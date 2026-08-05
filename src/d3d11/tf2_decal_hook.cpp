#include "tf2_decal_hook.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <windows.h>

#include "../util/log/log.h"
#include "../util/util_string.h"

// NV-DXVK [JobProbe]: the accumulators the worker hook writes into. Defined in
// dxvk/rtx_render/rtx_camera_manager.cpp, alongside g_engineHookCaptureCount,
// and drained once per frame by InstanceManager. They live on that side of the
// layer boundary because the reader compiles into libdxvk.a, which dxgi.dll
// links with no d3d11 objects — see the comment at the definition.
namespace dxvk { namespace tf2 {
  extern std::atomic<uint64_t> g_jobProbeCalls;
  extern std::atomic<uint64_t> g_jobProbeRecCntSum;
  extern std::atomic<uint64_t> g_jobProbeBadReads;
  extern std::atomic<uint32_t> g_jobProbeJobIdxMin;
  extern std::atomic<uint32_t> g_jobProbeJobIdxMax;

  // NV-DXVK [DrainProbe]: same arrangement, for the worker's output.
  extern std::atomic<uint64_t> g_drainProbeCalls;
  extern std::atomic<uint64_t> g_drainProbeBad;
  extern std::atomic<uint64_t> g_drainProbeM1Sum;
  extern std::atomic<uint64_t> g_drainProbeM2Sum;
  extern std::atomic<uint64_t> g_drainProbeRSum;
  extern std::atomic<uint32_t> g_drainProbeM1Max;
  extern std::atomic<uint32_t> g_drainProbeM2Max;
  extern std::atomic<uint32_t> g_drainProbeC70;
  extern std::atomic<uint32_t> g_drainProbeC74;
  extern std::atomic<uint32_t> g_drainProbeC78;
  extern std::atomic<uint32_t> g_drainProbeLayoutOk;
}}

namespace tf2_decal_hook {

  // ===== TLS counter ======================================================
  // We use a thread-local DEPTH COUNTER (not a flag) so that:
  //   - Multiple draw threads work independently (TF2 issues decals from
  //     several render-job threads in parallel).
  //   - Re-entrancy from the dispatcher (sub_1801B11B0 walking the command
  //     buffer) into a nested decal-emit is handled correctly.
  // The detour increments on entry and decrements on the unique return
  // point at the bottom of the wrapper; multi-return cleanup is implicit
  // because we wrap the entire original-function call.
  thread_local int t_decalDepth = 0;

  bool IsInDecalRender() {
    return t_decalDepth > 0;
  }

  // ===== Hook state =======================================================
  namespace {
    // Function pointer the wrapper calls — points into the trampoline page
    // we allocated, which executes the saved prologue then JMPs back to
    // engine.dll!sub_1801B4330+kPrologueSize.
    using DecalRenderFn = std::int64_t(__fastcall*)(std::int64_t a1,
                                                    std::uint32_t* a2,
                                                    int a3);
    DecalRenderFn s_origDecalRender = nullptr;

    // ===== Signature ======================================================
    // AOB pattern for sub_1801B4330 prologue. Captured from IDA at
    // engine.dll RVA 0x1B4330 (preferred base 0x180000000, so
    // 0x1801B4330 - 0x180000000 = 0x1B4330; an earlier revision of this
    // comment said 0xB4330, which is short by 0x100000 — the scan below
    // never used it, but anything computing a base from it would land
    // 1 MB off):
    //   44 89 44 24 18     mov [rsp+18], r8d         ; param spill (3rd arg)
    //   48 89 54 24 10     mov [rsp+10], rdx         ; param spill (2nd arg)
    //   56                 push rsi
    //   41 55              push r13
    //   B8 C8 80 00 00     mov eax, 0x80C8           ; stack frame size — distinctive
    //   E8 ?? ?? ?? ??     call __alloca_probe       ; rel32 (wildcarded)
    //   48 2B E0           sub rsp, rax
    //   48 8B F2           mov rsi, rdx
    //   45 8B E8           mov r13d, r8d
    //
    // The mov-immediate 0x80C8 is the unique fingerprint — a 32968-byte stack
    // frame on a 3-arg __alloca_probe'd function is not going to collide
    // anywhere else. Pattern length 32 bytes, single wildcard run for the
    // rel32 inside the call.
    constexpr std::size_t kPatternLen = 32;
    static constexpr std::uint8_t kPattern[kPatternLen] = {
      0x44, 0x89, 0x44, 0x24, 0x18,
      0x48, 0x89, 0x54, 0x24, 0x10,
      0x56,
      0x41, 0x55,
      0xB8, 0xC8, 0x80, 0x00, 0x00,
      0xE8, 0x00, 0x00, 0x00, 0x00,
      0x48, 0x2B, 0xE0,
      0x48, 0x8B, 0xF2,
      0x45, 0x8B, 0xE8,
    };
    // 0 = match required, 1 = wildcard (rel32 of the alloca_probe call).
    static constexpr std::uint8_t kMask[kPatternLen] = {
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0,
      0, 0,
      0, 0, 0, 0, 0,
      0, 1, 1, 1, 1,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
    };

    // Number of bytes we overwrite at the function entry. We need at least
    // 14 (for a `JMP [RIP+0]; <8-byte abs target>` indirect jump). The
    // smallest instruction boundary in the prologue at or above 14 is at
    // 18 bytes — the end of `mov eax, 0x80C8`. We copy those 18 bytes to
    // the trampoline. None of the five instructions in that span are
    // RIP-relative or position-dependent (no `jmp`/`call`/`mov [rip+x]`),
    // so the trampoline copy executes correctly at its new address.
    constexpr std::size_t kPrologueSize = 18;
    // Detour stub: `JMP [RIP+0]; <8-byte abs target>` = 14 bytes total.
    constexpr std::size_t kJmpStubSize = 14;

    // ===== Pattern scanner ================================================
    // Linear AOB scan over the module's .text section. engine.dll is ~349MB
    // total but .text is much smaller; we walk the PE section table to find
    // it and scan only that. Returns nullptr on no-match.
    std::uint8_t* ScanForPattern(HMODULE engine) {
      auto base = reinterpret_cast<std::uint8_t*>(engine);
      auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return nullptr;
      auto nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
      if (nt->Signature != IMAGE_NT_SIGNATURE)
        return nullptr;

      const auto* section = IMAGE_FIRST_SECTION(nt);
      const std::uint8_t* textBegin = nullptr;
      std::size_t textSize = 0;
      for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        if (std::memcmp(section->Name, ".text", 5) == 0) {
          textBegin = base + section->VirtualAddress;
          textSize = section->Misc.VirtualSize;
          break;
        }
      }
      if (textBegin == nullptr || textSize < kPatternLen)
        return nullptr;

      // Tight linear scan. ~30 MB / GHz cycles ~= sub-30ms one-shot.
      const std::uint8_t* const end = textBegin + textSize - kPatternLen;
      for (const std::uint8_t* p = textBegin; p <= end; ++p) {
        bool match = true;
        for (std::size_t k = 0; k < kPatternLen; ++k) {
          if (kMask[k] == 0 && p[k] != kPattern[k]) {
            match = false;
            break;
          }
        }
        if (match)
          return const_cast<std::uint8_t*>(p);
      }
      return nullptr;
    }

    // ===== [PropMask] pass-mask census ====================================
    //
    // Question this answers: of the three culls inside sub_1801B4330, which
    // one drops the meshes? The distance cull is already OUT -- meshes vanish
    // on frames with camMoved=0 (f=9967: 12 gone, camera bit-identical), and
    // no distance threshold can be crossed when no distance changes. That
    // leaves the pass mask (a3 & propFlagByte) and the per-entry predicate
    // (entry->vtbl[152]).
    //
    // We cannot see inside the function from a wrapper, but we do not need
    // to: a2 and a3 are exactly the inputs the mask test uses, so walking
    // them here reproduces cull 1 faithfully.
    //
    // Layout, read off the decompile of sub_1801B4330 and cross-checked
    // against sub_1801B36D0 (both index the same arrays):
    //   a2[8197]        end index of the pass-3 entry range
    //   a2[8199]        start index of the pass-3 entry range
    //   a2 + 8202       entry array, 4 dwords (16B) per entry
    //     entry[13]     bitmask byte; 0 = entry contributes nothing
    //     u16 @ +8      first prop index
    //     u16 @ +10     one-past-last prop index
    //   a2 + 16394      per-prop records, 4 dwords (16B) each
    //     rec[0]        PROP FLAG BYTE -- the operand of `a3 & flags`
    //     rec[1]        bucket id (matched against the BitScanForward bit)
    //     u32 @ +4      transform index
    //     f32 @ +12     the distance value the (acquitted) distance cull used
    //
    // CORRECTED 2026-07-30. The first version read the flag byte at rec[3] and
    // measured maskFail == props on every single line, with passHash never
    // leaving its seed -- i.e. "no prop ever passes", which cannot be true or
    // the function would draw nothing at all. A census that says the game
    // renders nothing is measuring the wrong bytes, not finding a 100% cull.
    //
    // The error: in `*(_BYTE *)(v28 - 3)` the pointer v28 is a float*, so the
    // -3 is 3 FLOATS = 12 bytes, not 3 bytes. Working it through,
    //   v28      = &a2[4i + 16397] as float*  -> byte 4*(4i+16397)
    //   v28 - 3  = byte 4*(4i+16397) - 12     = byte 4*(4i+16394) = rec[0]
    // The other two passes agree once their own pointer types are respected:
    //   pass 3: v76 = &a2[4i+16395] as uint*, v76-1 -> -4 bytes -> rec[0]
    //   pass 2: v52 = &a2[4i+16395] as float*, v52-1 -> -4 bytes -> rec[0]
    // and all three put the bucket id at rec[1] (v28-11, v76-3, v52-3), which
    // is the byte the original decode did get right -- that agreement is what
    // makes the record base (4i + 16394) trustworthy while the offset was not.
    //
    // Self-check on that layout: the highest prop index the entries can name
    // is 16384, which puts the last record at dword 81930 -- immediately below
    // a2[81932]/a2[81939]/a2[81941], the flag and job handles the function
    // reads. It also matches the v104[16384] scratch array on its stack. Two
    // independent confirmations that 16394 + 4*i is right.
    // ===== [PropMask] upstream-state probe ================================
    //
    // The census established that NONE of the three culls inside
    // sub_1801B4330 drop these meshes: the pass mask rejected 0 of 543 props
    // across 2336 calls, and on burst frames the props are already absent
    // from the INPUT (p1/p2/p3 caving 256/541/24 -> 180/354/2 -> 99/291/39
    // with the camera stationary). So the interesting state is upstream of
    // the function, in whatever produced those counts.
    //
    // These are the engine globals the function reads to find its data.
    // Absolute addresses from IDA (preferred base 0x180000000):
    //   0x193B870F0  parity word; the function selects its transform buffer
    //                as qword_193F07A68[3 * (this & 1)] -- a DOUBLE BUFFER.
    //                A parity flipping out of step with the list build is the
    //                shape of "same viewpoint, wildly different list".
    //   0x193F07A50  base of the 112-byte-per-prop transform array (v101)
    //   0x193F07A68  base array of 80-byte-per-prop buffers, stride 3 qwords
    constexpr std::uintptr_t kRvaParity   = 0x13B870F0;
    constexpr std::uintptr_t kRvaXformA   = 0x13F07A50;
    constexpr std::uintptr_t kRvaXformSel = 0x13F07A68;

    // Read a qword out of engine.dll, or return 0 if anything is off.
    // Re-resolves the module every call rather than caching a foreign
    // module's base -- cached foreign pointers are how the CS-thread
    // quit-crash happened.
    std::uint64_t ReadEngineQword(std::uintptr_t rva) {
      HMODULE engine = GetModuleHandleA("engine.dll");
      if (engine == nullptr)
        return 0;
      const auto* p = reinterpret_cast<const std::uint8_t*>(engine) + rva;
      MEMORY_BASIC_INFORMATION mbi = {};
      if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT)
        return 0;
      if ((mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0)
        return 0;
      return *reinterpret_cast<const std::uint64_t*>(p);
    }

    // OFF 2026-07-30, kept intact rather than deleted.
    //
    // This census targets sub_1801B4330, the MAIN pass. A spread-sampled
    // [MeshTraceSite] run (3 per frame over 1000 frames, replacing a burst of
    // 48 that all landed in 3 adjacent frames) showed the traced meshes do not
    // come through the main pass at all:
    //     eng+0x1b3bc8  early depth prepass   2935 / 3000   97.8%
    //     mat-only path                          33 / 3000
    //     studio+0x11e91                         32 / 3000
    //     eng+0x1b4ad6  MAIN PASS                 0 / 3000
    // The original 48-sample burst had reported 39x main pass / 7x prepass --
    // the exact inverse. So everything this census measured is true, and is
    // about a function that essentially never draws these meshes.
    //
    // Kept because its SILENCE is the finding: 0 mask rejections over 543 props
    // x 2336 calls is what acquits the main pass. If the prepass probe also
    // comes back clean, re-enable this rather than rebuilding it from scratch.
    constexpr bool kLogPassMaskCensus = false;
    constexpr int  kPassMaskMaxLines  = 4000;
    constexpr std::uint32_t kMaxPropIndex = 16384;

    std::atomic<int> s_passMaskLines{ 0 };

    void LogPassMaskCensus(const std::uint32_t* a2, int a3, const char* when) {
      if (!kLogPassMaskCensus || a2 == nullptr)
        return;
      if (s_passMaskLines.load(std::memory_order_relaxed) >= kPassMaskMaxLines)
        return;

      const std::uint32_t endEnt   = a2[8197];
      const std::uint32_t cnt8198  = a2[8198];
      const std::uint32_t startEnt = a2[8199];
      const std::uint32_t cnt8200  = a2[8200];

      // Fail closed on anything that does not look like the shape we decoded,
      // rather than walking arbitrary memory on a different engine build.
      if (endEnt > 65536u || startEnt > endEnt)
        return;

      const std::uint8_t passMask = static_cast<std::uint8_t>(a3);

      // Walk ONE pass's entry range. The function has three, over two
      // different entry arrays, and the first version of this census only
      // covered pass 3 -- about a quarter of the entries. That was enough to
      // kill the pass-mask hypothesis (it never rejected anything in 1185
      // calls) but NOT enough to call the function innocent, since a drop
      // could be sitting in either unmeasured pass.
      //   pass 1: array a2+12298, range [0, a2[8198])
      //   pass 2: array a2+8202,  range [0, a2[8199])
      //   pass 3: array a2+8202,  range [a2[8199], a2[8197])
      // Entry layout and per-prop record layout are identical in all three
      // (verified against each pass's own decompile).
      struct PassStat {
        std::uint32_t ents = 0, props = 0, fail = 0;
        std::uint64_t hash = 1469598103934665603ull;
      };

      auto walkPass = [&](const std::uint32_t* entries,
                          std::uint32_t first, std::uint32_t lastEnt,
                          PassStat& st, std::uint8_t* rawOut, bool* rawSet) {
        if (lastEnt > 65536u || first > lastEnt)
          return;
        for (std::uint32_t e = first; e < lastEnt; ++e) {
          const auto* ent =
            reinterpret_cast<const std::uint8_t*>(entries + 4ull * e);
          if (ent[13] == 0)
            continue;
          const std::uint16_t p0 = *reinterpret_cast<const std::uint16_t*>(ent + 8);
          const std::uint16_t p1 = *reinterpret_cast<const std::uint16_t*>(ent + 10);
          if (p1 < p0 || p1 > kMaxPropIndex)
            continue;
          ++st.ents;
          for (std::uint32_t i = p0; i < p1; ++i) {
            const auto* rec =
              reinterpret_cast<const std::uint8_t*>(a2 + 16394ull + 4ull * i);
            if (rawOut != nullptr && !*rawSet) {
              for (int b = 0; b < 16; ++b) rawOut[b] = rec[b];
              *rawSet = true;
            }
            ++st.props;
            if ((passMask & rec[0]) == 0) {
              ++st.fail;
            } else {
              st.hash = (st.hash ^ i) * 1099511628211ull;
              st.hash = (st.hash ^ rec[0]) * 1099511628211ull;
            }
          }
        }
      };

      PassStat p1s, p2s, p3s;
      std::uint32_t entsWalked = 0, props = 0, maskFail = 0;
      // Raw bytes of the FIRST record we walk. The whole census is only as
      // good as the layout decode, so print the bytes it is decoding rather
      // than making the counts the only evidence -- the rec[3]/rec[0] mistake
      // was invisible in the counts alone (they were self-consistently wrong)
      // and would have been obvious here.
      std::uint8_t rawRec[16] = {};
      bool haveRaw = false;
      // FNV-1a over the props that PASS. If the camera is still and this hash
      // changes between calls, the surviving set is churning -- which is the
      // drop mechanism, visible directly.
      std::uint64_t passHash = 1469598103934665603ull;

      walkPass(a2 + 12298, 0u,        cnt8198,  p1s, rawRec, &haveRaw);
      walkPass(a2 + 8202,  0u,        startEnt, p2s, nullptr, nullptr);
      walkPass(a2 + 8202,  startEnt,  endEnt,   p3s, nullptr, nullptr);

      entsWalked = p1s.ents  + p2s.ents  + p3s.ents;
      props      = p1s.props + p2s.props + p3s.props;
      maskFail   = p1s.fail  + p2s.fail  + p3s.fail;
      // Fold the three per-pass hashes together so one number still answers
      // "did the surviving set change at all".
      passHash = ((passHash ^ p1s.hash) * 1099511628211ull);
      passHash = ((passHash ^ p2s.hash) * 1099511628211ull);
      passHash = ((passHash ^ p3s.hash) * 1099511628211ull);

      // Upstream state, sampled at the same instant as the counts above so a
      // collapse can be lined up against it directly.
      //
      // MEASURED 2026-07-30: a2[81932] is NOT a boolean enable, despite the
      // decompile reading `if (v3[81932])`. It is a COUNT, and it predicts the
      // total prop count this function sees at pearson r = 0.978 over 1130
      // samples (gate=14 -> props=16; gate=462 -> props=543). So the props are
      // not culled anywhere inside this function -- it is simply handed a
      // shorter list, and this field is that list's length.
      //
      // ANSWERED: exactly one site in engine.dll writes it --
      //   0x1801B28A9   mov [rdi+50030h], edx   in sub_1801B2200
      // and sub_1801B2200 is GatherVisibleStaticProps, named with certainty
      // from a cvar it reads (staticProp_GatherVisibleStaticProps_Yield at
      // 0x193B872F8). Every other reference in the static-prop code only READS
      // the field -- the three draw commands (sub_1801B31E0, the prepass and
      // the main pass) just consume the list it produces.
      //
      // So the real per-object visibility test lives in the gather, not in any
      // draw function: a per-prop distance/fade compare,
      //   dist^2 / (a1+327752)^2  <=  propRadius^2 * f(model_fadeRangeFraction)
      // with r_lod_switch_scale and a runtime value from
      // qword_194C657B0->vtbl[344] also in the threshold. Props that fail are
      // never appended, so they are gone before any draw function sees them --
      // which is exactly why every cull INSIDE those functions measured clean
      // (0 mask rejections over 543 props x 2336 calls).
      //
      // Note for anyone re-deriving this: IDA's immediate search does NOT match
      // displacement operands, so find(type="immediate", 327756) returns
      // nothing even though `mov ecx,[rsi+5004Ch]` plainly exists. Scan o_displ
      // operand VALUES over a BOUNDED range instead -- walking all 6 MB of
      // .text in a script hangs the plugin server.
      const std::uint32_t gate = a2[81932];
      const std::uint32_t job1 = a2[81939];   // job waited on at entry
      const std::uint32_t job2 = a2[81941];   // second job (12288 = "none")
      const std::uint64_t parityRaw = ReadEngineQword(kRvaParity);
      const std::uint64_t parity    = parityRaw & 1ull;
      // Inputs to GatherVisibleStaticProps' visibility test. Its a1 IS this
      // a2 -- it writes +327728 and these functions read it -- so the test's
      // operands are reachable from here without hooking the gather at all:
      //   +327740/744/748 = the camera position the GATHER used (a copy, not
      //                     necessarily the live camera)
      //   +327752         = the scale the squared distance is divided by
      // The test is dist^2/scale^2 <= propRadius^2 * f(model_fadeRangeFraction),
      // so a prop can cross it with the camera STATIONARY if the threshold
      // moves. That is the only way to reconcile "the cull is distance-based"
      // with "meshes drop at camMoved=0", and these fields decide it: if
      // gScale or the gather's camera copy jitters while the real camera is
      // still, the threshold is moving under the props.
      const float gCamX  = *reinterpret_cast<const float*>(a2 + 81935);
      const float gCamY  = *reinterpret_cast<const float*>(a2 + 81936);
      const float gCamZ  = *reinterpret_cast<const float*>(a2 + 81937);
      const float gScale = *reinterpret_cast<const float*>(a2 + 81938);
      const std::uint64_t xformA    = ReadEngineQword(kRvaXformA);
      // qword_193F07A68[3 * parity] -- the buffer actually selected this call.
      const std::uint64_t xformSel  =
        ReadEngineQword(kRvaXformSel + 8ull * 3ull * parity);

      std::string raw;
      if (haveRaw) {
        static const char* kHex = "0123456789abcdef";
        for (int b = 0; b < 16; ++b) {
          raw += kHex[rawRec[b] >> 4];
          raw += kHex[rawRec[b] & 0xF];
          if (b == 3 || b == 7 || b == 11)
            raw += '_';
        }
      }

      s_passMaskLines.fetch_add(1, std::memory_order_relaxed);
      dxvk::Logger::warn(dxvk::str::format(
        "[PropMask] when=", when,
        " a3=0x", std::hex, static_cast<std::uint32_t>(a3),
        " mask=0x", static_cast<std::uint32_t>(passMask), std::dec,
        " ents=[", startEnt, ",", endEnt, ") c8198=", cnt8198,
        " c8200=", cnt8200,
        " entsWalked=", entsWalked,
        " props=", props,
        " maskFail=", maskFail,
        // Per-pass breakdown: a drop confined to one pass is invisible in the
        // totals, which is exactly how pass 1 and 2 went unmeasured before.
        " p1=", p1s.props, "/", p1s.fail,
        " p2=", p2s.props, "/", p2s.fail,
        " p3=", p3s.props, "/", p3s.fail,
        " passHash=0x", std::hex, passHash, std::dec,
        // Upstream state. If parity/xformSel move in lockstep with a p1/p2/p3
        // collapse, the double buffer is the mechanism. If all of these are
        // steady while the counts cave, the fault is inside the job itself.
        " gate=", gate,
        " gCam=(", gCamX, ",", gCamY, ",", gCamZ, ")",
        " gScale=", gScale,
        " job1=", job1,
        " job2=", job2,
        " parity=", static_cast<std::uint32_t>(parity),
        " xformA=0x", std::hex, xformA,
        " xformSel=0x", xformSel, std::dec,
        " rec0=", raw.empty() ? "-" : raw.c_str()));
    }

    // ===== Detour wrapper =================================================
    // Increments the TLS counter, calls the original (via trampoline),
    // decrements, returns the result. Single return point — RAII would
    // also work but adds noise; explicit dec is clearer here.
    std::int64_t __fastcall DecalRenderWrapper(std::int64_t a1,
                                               std::uint32_t* a2,
                                               int a3) {
      // SEH-safe counter management. If engine.dll's sub_1801B4330 ever
      // raises a structured exception (job-system aborts, asserts), a
      // bare increment/decrement pair would leak the counter into the
      // next call on this thread — every subsequent draw would then
      // wrongly classify as decal. __finally runs during SEH unwind,
      // unlike C++ destructors. The cost is one TEB-relative entry on
      // the exception chain per call, which is negligible compared to
      // the work the wrapped function does.
      ++t_decalDepth;
      std::int64_t r = 0;
      __try {
        // [PropMask] census BEFORE and AFTER, because the ordering inside the
        // wrapped function makes "before" alone untrustworthy:
        //
        //   JT_WaitForJob(a2[81939]);        // = 12288, the "no job" sentinel
        //   if (a2[81932]) {                 // reads the COUNT here...
        //     if (a2[81941] != 12288)
        //       JT_WaitForJob(a2[81941]);    // ...and only NOW waits for the
        //   }                                //    GatherVisibleStaticProps job
        //                                    //    that produces it
        //
        // GatherVisibleStaticProps (sub_1801B2200, named from its own cvar
        // staticProp_GatherVisibleStaticProps_Yield) writes that count at its
        // very END. So a census taken at wrapper entry -- before both waits --
        // can be sampling the gather MID-WALK rather than the finished list.
        // The first version of this probe did exactly that, which means its
        // reported "input collapses" could have been mid-flight state.
        //
        // RESOLVED 2026-07-30 by censusing both sides and diffing: pre == post
        // on ALL 2000 pairs -- identical props, per-pass breakdown, passHash
        // and gate. The list is fully settled before the draw function runs,
        // there is no job-in-flight race, and the collapses measured by the
        // earlier entry-only probe were REAL rather than artifacts of reading
        // mid-walk. The "post" census is therefore dropped: it costs half the
        // line budget to re-confirm a settled question. Restore it (and the
        // when= field) if the wait ordering above ever changes.
        LogPassMaskCensus(a2, a3, "pre");
        r = s_origDecalRender(a1, a2, a3);
      } __finally {
        --t_decalDepth;
      }
      return r;
    }

    // ===== Trampoline allocation ==========================================
    // Allocate an executable page near the target so the JMP-back from the
    // trampoline to engine.dll+kPrologueSize can use a 32-bit rel32 if
    // desired (we use absolute jmp here for simplicity, so proximity is
    // not strictly required — but allocating near keeps the address space
    // tidy).
    std::uint8_t* AllocateTrampolinePage() {
      // 4 KB is plenty for 18 bytes + 14-byte JMP back. VirtualAlloc
      // returns 64KB-aligned regions; RWX is acceptable here as this page
      // never holds user data.
      auto* p = static_cast<std::uint8_t*>(VirtualAlloc(
          nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
      return p;
    }

    // ===== PREPASS hook: sub_1801B36D0 ====================================
    //
    // This is where the traced meshes actually come from -- 2935 of 3000
    // spread-sampled [MeshTraceSite] emissions resolve to eng+0x1b3bc8, the
    // submit inside StaticPropMgr_DrawCmd_EarlyDepthPrepass, versus ZERO for
    // the main pass this file's other hook targets.
    //
    // Signature is (a1, a2) -- TWO args. There is no a3 pass mask, so the
    // main-pass census does not apply. The prepass culls are its own:
    //   1. entry[15] flag, OR staticProp_earlyDepthPrepassIncludeOpaques
    //   2. outer `break` at 0x1801B38D6 once a prop exceeds
    //      fmax(earlyDepthPrepassDist^2, includeOpaquesDist^2) -- terminates
    //      the WHOLE remaining run, not one prop
    //   3. inner `while (propDist <= limit)` at 0x1801B3AF8, where limit is
    //      earlyDepthPrepassDist^2 if entry[15] else includeOpaquesDist^2
    //   4. bucket-id mismatch ending a run early
    // So the measurement is: props CLAIMED by the entry ranges vs props that
    // actually clear the distance limit. A gap is the prepass dropping them.
    constexpr bool kLogPrepassCensus   = true;
    constexpr int  kPrepassMaxLines    = 4000;
    // ConVar object RVAs (value float at +0x58, int at +0x5C; see tf2_engine_cvars).
    constexpr std::uintptr_t kRvaCvPrepassDist    = 0x13B87470;
    constexpr std::uintptr_t kRvaCvOpaquesDist    = 0x13B87110;
    constexpr std::uintptr_t kRvaCvIncludeOpaques = 0x13B87350;

    std::atomic<int> s_prepassLines{ 0 };

    float ReadEngineCvarFloat(std::uintptr_t objRva) {
      HMODULE engine = GetModuleHandleA("engine.dll");
      if (engine == nullptr) return -1.0f;
      const auto* obj = reinterpret_cast<const std::uint8_t*>(engine) + objRva;
      MEMORY_BASIC_INFORMATION mbi = {};
      if (VirtualQuery(obj, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT)
        return -1.0f;
      const auto* parent = *reinterpret_cast<const std::uint8_t* const*>(obj + 0x38);
      if (parent == nullptr) parent = obj;
      if (VirtualQuery(parent, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT)
        return -1.0f;
      return *reinterpret_cast<const float*>(parent + 0x58);
    }

    void LogPrepassCensus(const std::uint32_t* a2) {
      if (!kLogPrepassCensus || a2 == nullptr) return;
      if (s_prepassLines.load(std::memory_order_relaxed) >= kPrepassMaxLines) return;

      const std::uint32_t endEnt   = a2[8197];
      const std::uint32_t startEnt = a2[8199];
      const std::uint32_t cnt8198  = a2[8198];
      const std::uint32_t gate     = a2[81932];
      if (endEnt > 65536u || startEnt > endEnt) return;

      const float dPrepass = ReadEngineCvarFloat(kRvaCvPrepassDist);
      const float dOpaques = ReadEngineCvarFloat(kRvaCvOpaquesDist);
      const float incOpq   = ReadEngineCvarFloat(kRvaCvIncludeOpaques);
      const float limA = dPrepass * dPrepass;                 // entry[15] set
      const float limB = (incOpq != 0.0f) ? dOpaques * dOpaques : 0.0f;
      const float limOuter = (limA > limB) ? limA : limB;

      std::uint32_t claimed = 0, passDist = 0, failDist = 0, ents = 0, skipped = 0;
      std::uint32_t firstOverIdx = 0xFFFFFFFFu;
      for (std::uint32_t e = 0; e < endEnt; ++e) {
        const auto* ent = reinterpret_cast<const std::uint8_t*>(a2 + 8202ull + 4ull * e);
        // cull 1: entry contributes only if its flag byte is set OR opaques included
        if (ent[15] == 0 && incOpq == 0.0f) { ++skipped; continue; }
        if (ent[13] == 0) { ++skipped; continue; }
        const std::uint16_t p0 = *reinterpret_cast<const std::uint16_t*>(ent + 8);
        const std::uint16_t p1 = *reinterpret_cast<const std::uint16_t*>(ent + 10);
        if (p1 < p0 || p1 > kMaxPropIndex) { ++skipped; continue; }
        ++ents;
        const float lim = ent[15] ? limA : limB;
        for (std::uint32_t i = p0; i < p1; ++i) {
          const auto* rec = reinterpret_cast<const std::uint8_t*>(a2 + 16394ull + 4ull * i);
          const float d = *reinterpret_cast<const float*>(rec + 12);
          ++claimed;
          if (d <= lim) ++passDist;
          else { ++failDist; if (firstOverIdx == 0xFFFFFFFFu) firstOverIdx = i; }
        }
      }

      s_prepassLines.fetch_add(1, std::memory_order_relaxed);
      dxvk::Logger::warn(dxvk::str::format(
        "[PrepassMask] ents=[0,", endEnt, ") start=", startEnt,
        " c8198=", cnt8198, " gate=", gate,
        " entsUsed=", ents, " entsSkipped=", skipped,
        " claimed=", claimed,
        " passDist=", passDist,
        " failDist=", failDist,
        " firstOver=", firstOverIdx,
        " limA=", limA, " limB=", limB, " limOuter=", limOuter,
        " cvPrepassDist=", dPrepass, " cvOpaquesDist=", dOpaques,
        " cvIncOpaques=", incOpq));
    }

    // Prologue of sub_1801B36D0, read from IDA:
    //   48 89 54 24 10        mov [rsp+10h], rdx      (5)
    //   55                    push rbp                (1)  -> 6
    //   48 81 EC F0 00 00 00  sub rsp, 0F0h           (7)  -> 13
    //   48 8B EA              mov rbp, rdx            (3)  -> 16
    //   49 83 C8 FF           or r8, -1               (4)
    //   33 D2                 xor edx, edx            (2)
    // 16 bytes is the first instruction boundary at or above the 14 a
    // `JMP [RIP+0]; <abs>` needs, and none of those four are RIP-relative or
    // position-dependent, so the trampoline copy runs correctly relocated.
    constexpr std::size_t kPrepassPatternLen  = 22;
    constexpr std::size_t kPrepassPrologueSize = 16;
    static constexpr std::uint8_t kPrepassPattern[kPrepassPatternLen] = {
      0x48, 0x89, 0x54, 0x24, 0x10,
      0x55,
      0x48, 0x81, 0xEC, 0xF0, 0x00, 0x00, 0x00,
      0x48, 0x8B, 0xEA,
      0x49, 0x83, 0xC8, 0xFF,
      0x33, 0xD2,
    };

    using PrepassFn = std::int64_t(__fastcall*)(std::int64_t a1, std::uint32_t* a2);
    PrepassFn s_origPrepass = nullptr;

    std::int64_t __fastcall PrepassWrapper(std::int64_t a1, std::uint32_t* a2) {
      std::int64_t r = 0;
      __try {
        LogPrepassCensus(a2);
        r = s_origPrepass(a1, a2);
      } __finally {
      }
      return r;
    }

    // Generic AOB scan over a module's .text for an exact (no-wildcard) pattern.
    std::uint8_t* ScanExact(HMODULE mod, const std::uint8_t* pat, std::size_t len) {
      auto base = reinterpret_cast<std::uint8_t*>(mod);
      auto dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
      if (dos->e_magic != IMAGE_DOS_SIGNATURE) return nullptr;
      auto nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
      if (nt->Signature != IMAGE_NT_SIGNATURE) return nullptr;
      const auto* sec = IMAGE_FIRST_SECTION(nt);
      const std::uint8_t* begin = nullptr; std::size_t size = 0;
      for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec) {
        if (std::memcmp(sec->Name, ".text", 5) == 0) {
          begin = base + sec->VirtualAddress; size = sec->Misc.VirtualSize; break;
        }
      }
      if (begin == nullptr || size < len) return nullptr;
      for (const std::uint8_t* p = begin; p <= begin + size - len; ++p)
        if (std::memcmp(p, pat, len) == 0)
          return const_cast<std::uint8_t*>(p);
      return nullptr;
    }

    // Install the prepass detour. Independent of the main-pass hook's success:
    // that one is now census-disabled and only feeds IsInDecalRender, while
    // THIS is where the traced meshes actually get emitted.
    void InstallPrepassHook(HMODULE engine) {
      if (!kLogPrepassCensus) return;
      std::uint8_t* target = ScanExact(engine, kPrepassPattern, kPrepassPatternLen);
      if (target == nullptr) {
        dxvk::Logger::warn(
          "[tf2_prepass_hook] sub_1801B36D0 prologue NOT FOUND -- prepass "
          "census disabled. Engine build mismatch, or already patched.");
        return;
      }
      std::uint8_t* tramp = AllocateTrampolinePage();
      if (tramp == nullptr) return;

      std::memcpy(tramp, target, kPrepassPrologueSize);
      std::uint8_t* jmpBack = tramp + kPrepassPrologueSize;
      jmpBack[0] = 0xFF; jmpBack[1] = 0x25;
      jmpBack[2] = jmpBack[3] = jmpBack[4] = jmpBack[5] = 0x00;
      auto absBack = reinterpret_cast<std::uintptr_t>(target + kPrepassPrologueSize);
      std::memcpy(jmpBack + 6, &absBack, sizeof(absBack));
      s_origPrepass = reinterpret_cast<PrepassFn>(tramp);

      std::uint8_t patch[kPrepassPrologueSize];
      std::memset(patch, 0x90, sizeof(patch));   // NOP tail
      patch[0] = 0xFF; patch[1] = 0x25;
      patch[2] = patch[3] = patch[4] = patch[5] = 0x00;
      auto absWrap = reinterpret_cast<std::uintptr_t>(&PrepassWrapper);
      std::memcpy(patch + 6, &absWrap, sizeof(absWrap));

      DWORD oldProt = 0;
      if (!VirtualProtect(target, kPrepassPrologueSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn("[tf2_prepass_hook] VirtualProtect failed -- not installed.");
        s_origPrepass = nullptr;
        return;
      }
      std::memcpy(target, patch, kPrepassPrologueSize);
      DWORD tmp = 0;
      VirtualProtect(target, kPrepassPrologueSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, kPrepassPrologueSize);

      dxvk::Logger::info(dxvk::str::format(
        "[tf2_prepass_hook] Installed. target=0x", std::hex,
        reinterpret_cast<std::uintptr_t>(target),
        " trampoline=0x", reinterpret_cast<std::uintptr_t>(tramp),
        " wrapper=0x", absWrap, std::dec));
    }

    // ===== Install ========================================================
    // Returns true iff the hook is now live.
    bool DoInstall() {
      HMODULE engine = GetModuleHandleA("engine.dll");
      if (engine == nullptr) {
        dxvk::Logger::warn(
          "[tf2_decal_hook] engine.dll not loaded yet, skipping install.");
        return false;
      }

      std::uint8_t* target = ScanForPattern(engine);
      if (target == nullptr) {
        dxvk::Logger::warn(
          "[tf2_decal_hook] sub_1801B4330 prologue pattern NOT FOUND in "
          "engine.dll .text. Hook NOT installed (will fall back to no-op). "
          "Probable cause: engine.dll version mismatch or another mod has "
          "already patched the function entry.");
        return false;
      }
      dxvk::Logger::info(dxvk::str::format(
        "[tf2_decal_hook] Located sub_1801B4330 at 0x", std::hex,
        reinterpret_cast<std::uintptr_t>(target), std::dec));

      // Allocate trampoline page. Layout will be:
      //   [0..kPrologueSize)        copy of original prologue bytes
      //   [kPrologueSize..+14)      JMP [RIP+0]; <abs target = engine+kPrologueSize>
      std::uint8_t* tramp = AllocateTrampolinePage();
      if (tramp == nullptr) {
        dxvk::Logger::warn(
          "[tf2_decal_hook] VirtualAlloc for trampoline failed, aborting.");
        return false;
      }

      // Copy original prologue bytes verbatim. None of them are RIP-rel.
      std::memcpy(tramp, target, kPrologueSize);

      // Append `JMP [RIP+0]; <abs target>` returning into the original
      // function past the patched bytes.
      // Bytes: FF 25 00 00 00 00  <8-byte absolute address>
      std::uint8_t* jmpBack = tramp + kPrologueSize;
      jmpBack[0] = 0xFF;
      jmpBack[1] = 0x25;
      jmpBack[2] = 0x00;
      jmpBack[3] = 0x00;
      jmpBack[4] = 0x00;
      jmpBack[5] = 0x00;
      auto absTarget = reinterpret_cast<std::uintptr_t>(target + kPrologueSize);
      std::memcpy(jmpBack + 6, &absTarget, sizeof(absTarget));

      // s_origDecalRender = entry of the trampoline page.
      s_origDecalRender = reinterpret_cast<DecalRenderFn>(tramp);

      // Now overwrite the function entry with a JMP to our wrapper. Use
      // the same 14-byte indirect-jmp form so we don't clobber a register
      // at function entry (matters: at function entry, RCX/RDX/R8/R9 are
      // live with the arguments; clobbering RAX is technically fine in
      // Win64 but cleaner to avoid).
      std::uint8_t patchBytes[kPrologueSize] = {};
      patchBytes[0] = 0xFF;
      patchBytes[1] = 0x25;
      patchBytes[2] = 0x00;
      patchBytes[3] = 0x00;
      patchBytes[4] = 0x00;
      patchBytes[5] = 0x00;
      auto absWrapper = reinterpret_cast<std::uintptr_t>(&DecalRenderWrapper);
      std::memcpy(patchBytes + 6, &absWrapper, sizeof(absWrapper));
      // Pad the remaining 4 bytes with NOP so disassemblers stay aligned
      // and any debug-time single-step lands on a sane instruction.
      patchBytes[14] = 0x90;
      patchBytes[15] = 0x90;
      patchBytes[16] = 0x90;
      patchBytes[17] = 0x90;

      DWORD oldProtect = 0;
      if (!VirtualProtect(target, kPrologueSize, PAGE_EXECUTE_READWRITE,
                          &oldProtect)) {
        dxvk::Logger::warn(dxvk::str::format(
          "[tf2_decal_hook] VirtualProtect(RWX) failed at 0x", std::hex,
          reinterpret_cast<std::uintptr_t>(target), std::dec,
          " — hook NOT installed."));
        return false;
      }
      std::memcpy(target, patchBytes, kPrologueSize);
      DWORD tmp = 0;
      VirtualProtect(target, kPrologueSize, oldProtect, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, kPrologueSize);

      dxvk::Logger::info(dxvk::str::format(
        "[tf2_decal_hook] Installed. target=0x", std::hex,
        reinterpret_cast<std::uintptr_t>(target),
        " trampoline=0x", reinterpret_cast<std::uintptr_t>(tramp),
        " wrapper=0x", absWrapper, std::dec));

      InstallPrepassHook(engine);
      return true;
    }

    // ===== Install gate ===================================================
    std::once_flag s_onceInstall;
    std::atomic<bool> s_installed{false};
  }  // namespace

  bool EnsureInstalled() {
    std::call_once(s_onceInstall, []() {
      s_installed.store(DoInstall(), std::memory_order_release);
    });
    return s_installed.load(std::memory_order_acquire);
  }

  // =========================================================================
  // NV-DXVK [WorldVis] — find the PRODUCER of the per-view visibility bitmask.
  //
  // THE PROBLEM THIS SOLVES. Geometry disappears as the camera pitches down —
  // measured, not guessed: [PitchProbe] gives r = -0.934 between camera pitch
  // and live instance count, ~500 instances at 0 deg falling to ~284 at 50 deg,
  // monotonic. It is a frustum cull. It is NOT in client.dll: all four entry
  // points to the renderable frustum cull (sites 4a-4d) are patched dead and
  // the symptom is unchanged. Static-prop fade is off too, and unchanged.
  //
  // WHAT WE KNOW ABOUT THE TARGET. There is not one bitmask at +0x54088, nor
  // two. There are FIVE, packed back to back, and the earlier "world bitmask +
  // static-prop bitmask at +8*wordCount" model was wrong in a way that made the
  // measurement un-interpretable: it summed three of them into one number and
  // then correlated that number against one of its own components. The layout
  // is derived in full from sub_1802E7D10 (the allocator) above WvSnap; the
  // short version, in 64-bit words from +0x54088, with the five-DWORD header at
  // +0x54070:
  //     M1  [0,   c70)       world mask 1
  //     M2  [c70, c74)       static props -> StaticPropMgrClient005 vt+72/+80
  //     R   [c74, c78)       RENDERABLES  -> built by sub_1801A8350
  //     B   [c78, c78+c80)   leaf/PVS input -> read by sub_1801A8350
  //     T   [c7C, c7C+N)     second renderable mask
  //
  // WHY THAT CHANGES THE QUESTION. sub_1801A8350 is not a consumer of a mask
  // somebody else narrowed — it is the B -> R transform. Region B was measured
  // flat across every pitch bin, which was read as "the render-list build is
  // innocent". It is not: R, the output, had never been isolated. A flat input
  // and an unmeasured output is precisely the shape of a cull living inside the
  // transform, and the nine byte patches only killed the frustum call and the
  // fade inside it. What survives, all operating on R:
  //     R[i] &= *(leaf + 0x838 + 8i)   normal views      <-- wholesale AND
  //     R[i] &= *(leaf + 0xC38 + 8i)   (a2+36) & 0xC     <-- wholesale AND
  //     R[i] |= *(leaf + 0x438 + 8i)   force-visible OR
  //     bit cleared unless *(leaf + 16*idx + 0x1044) & 8
  //     a hard 4096-entry output cap that zeroes every remaining R word
  //
  // WHAT IT MEASURES. Popcounts of all five regions, R read AFTER the call (it
  // is zeroed before the job runs, so an entry-time read of R is 0 by
  // construction), plus the return value — the render-list entry count — and
  // the four ClientLeafSystem masks that feed R. Correlate against [PitchProbe]
  // on the same frame:
  //     rPost / ret FALL as pitch rises => the cull is inside sub_1801A8350,
  //       and andNormal/andShadow say whether it arrived via a leaf-system mask.
  //     rPost FLAT while instances fall => the loss is downstream of the render
  //       list entirely. Different fix, but no longer a guess either way.
  //
  // Cost: two popcount passes over ~730 words per call, plus a throttled log.
  // =========================================================================
  namespace {
    // RETARGETED 2026-08-04. The first version hooked engine.dll+0xB8000
    // (DrawWorldMeshesDepthOnly) because it was the only function that took the
    // bitmask's ADDRESS. Its own output disproved that choice: passMask was
    // 0x1003 on all 108 calls, `ctx` differed almost every call, and the whole
    // run produced ~2 calls/sec with worldBits averaging 76 of 8064 slots. That
    // is the cascaded-shadow-map depth pass (it is gated on
    // csm_world_shadow_meshes), sampled per-cascade, not per-frame — useless to
    // correlate against camera pitch.
    //
    // client.dll sub_1801A8350 (BuildRenderableRenderLists) is the right point,
    // but NOT for the reason the first retarget assumed. It was hooked as a
    // pure consumer of the world bitmask, sampled at entry. It is not a pure
    // consumer: it READS the leaf/PVS mask (region B, base ctx[0x54078], loaded
    // at 0x1801A84F8) and WRITES the renderable visibility mask (region R, base
    // ctx[0x54074], loaded at 0x1801A83D0). It is a B -> R transform, and every
    // cull that survived the nine byte patches lives between those two reads.
    //
    // Sampling at entry therefore could not see the output at all — R is zeroed
    // by sub_1802EF770/sub_1802EF510 before the job runs, so an entry-time
    // popcount of R reads ~0 by construction. The probe now snapshots the
    // inputs at entry, calls through, and reads R (and the return value, which
    // is the render-list entry count) on the way out.
    //
    // WHICH ARG IS THE CONTEXT is not statically obvious — the +0x54088 reads
    // use r8, but that is deep in the body and may be reloaded. So rather than
    // guess (guessing is what cost the last three rounds), all four integer
    // args are VALIDATED as candidate contexts and the probe reports which one
    // passed. Validation is now structural: the five-DWORD header at +0x54070
    // must be monotonic AND satisfy ctx[0x5407C] == ctx[0x54078] + ctx[0x54080],
    // which only holds for a context laid out by sub_1802E7D10.
    using WorldVisFn = std::int64_t(__fastcall*)(void*, void*, void*, void*);
    WorldVisFn s_origWorldVis = nullptr;

    constexpr std::uintptr_t kRvaWorldVisConsumer = 0x1A8350;   // client.dll sub_1801A8350

    // Context field offsets, from sub_1802E7D10 (the allocator — see the region
    // derivation above WvSnap). The engine.dll static-prop count that the old
    // probe used to size the prop region is deliberately gone: every region
    // length is carried in this five-DWORD header, so deriving one of them from
    // a separate global was what let the prop and world regions overlap.
    constexpr std::uintptr_t kOffBitmaskHeader = 0x54070;
    constexpr std::uintptr_t kOffWorldBitmask  = 0x54088;

    // sub_1801A8350 prologue:
    //   +0   4C 89 4C 24 20    mov [rsp+arg_18], r9   <-- 5 bytes, exactly a jmp
    //   +5   4C 89 44 24 18    mov [rsp+arg_10], r8
    //   +10  48 89 54 24 10    mov [rsp+arg_8], rdx
    //   ...
    //   +23  E8 ...            call __alloca_probe    <-- rel32, position dependent
    // The first instruction is 5 bytes on its own, which is exactly an
    // `E9 rel32`, so we steal only that one — no padding nops, and nothing
    // position-dependent copied (the alloca_probe call at +23 is far outside).
    constexpr std::size_t kWvPrologueSize = 5;

    bool WvReadable(const void* p, std::size_t len) {
      if (p == nullptr)
        return false;
      MEMORY_BASIC_INFORMATION mbi = {};
      if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT)
        return false;
      if ((mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0)
        return false;
      const auto start = reinterpret_cast<std::uintptr_t>(p);
      const auto end = reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
      return start + len <= end;
    }

    inline std::uint32_t WvPopcount64(std::uint64_t v) {
      // Written out rather than using an intrinsic so this cannot depend on
      // POPCNT being enabled in the build's target ISA.
      v = v - ((v >> 1) & 0x5555555555555555ull);
      v = (v & 0x3333333333333333ull) + ((v >> 2) & 0x3333333333333333ull);
      v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0Full;
      return static_cast<std::uint32_t>((v * 0x0101010101010101ull) >> 56);
    }

    // Allocate an RWX page within +/-2GB of `anchor` so a 5-byte rel32 jmp can
    // reach it. Walks outward from the module in 64KB steps.
    //
    // NB: the parameter is NOT called `near` — windows.h still defines `near`
    // and `far` as empty legacy macros, so `std::uint8_t* near` collapses to
    // `std::uint8_t*` and the declaration fails to parse.
    std::uint8_t* AllocateNearPage(std::uint8_t* anchor) {
      constexpr std::uintptr_t kStep = 0x10000;          // allocation granularity
      constexpr std::uintptr_t kRange = 0x40000000ull;   // 1GB each way, well inside rel32
      const auto base = reinterpret_cast<std::uintptr_t>(anchor);
      for (std::uintptr_t delta = kStep; delta < kRange; delta += kStep) {
        for (int dir = 0; dir < 2; ++dir) {
          const std::uintptr_t addr = dir ? (base + delta) : (base - delta);
          if (addr < kStep)
            continue;
          auto* p = static_cast<std::uint8_t*>(VirtualAlloc(
              reinterpret_cast<void*>(addr & ~(kStep - 1)), 4096,
              MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
          if (p != nullptr)
            return p;
        }
      }
      return nullptr;
    }

    std::atomic<std::uint64_t> s_wvCalls{0};

    // Resolve an address to module+RVA using the module's REAL SizeOfImage.
    //
    // The first version used a flat +/-512MB window per module, which
    // misattributed: the wrapper in d3d11.dll printed as
    // "client.dll+0x1859253d" — 408MB into a 64MB module — because d3d11.dll
    // happens to be mapped within 512MB above client.dll's base. A frame that
    // names the wrong module is worse than one that names none.
    bool WvModuleRange(const char* name, std::uintptr_t addr, std::uintptr_t& rvaOut) {
      HMODULE m = GetModuleHandleA(name);
      if (m == nullptr)
        return false;
      const auto base = reinterpret_cast<std::uintptr_t>(m);
      if (addr < base)
        return false;
      auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(m);
      if (!WvReadable(dos, sizeof(IMAGE_DOS_HEADER)) || dos->e_magic != IMAGE_DOS_SIGNATURE)
        return false;
      auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
        reinterpret_cast<const std::uint8_t*>(m) + dos->e_lfanew);
      if (!WvReadable(nt, sizeof(IMAGE_NT_HEADERS64)) || nt->Signature != IMAGE_NT_SIGNATURE)
        return false;
      const std::uintptr_t size = nt->OptionalHeader.SizeOfImage;
      if (addr - base >= size)
        return false;
      rvaOut = addr - base;
      return true;
    }

    // Popcount `words` 64-bit words at `base`, or return false if unreadable.
    bool WvPopRegion(std::uintptr_t base, std::uint32_t words, std::uint32_t& bitsOut) {
      if (words == 0 || words > 8192)
        return false;
      const auto* p = reinterpret_cast<const std::uint64_t*>(base);
      if (!WvReadable(p, sizeof(std::uint64_t) * words))
        return false;
      std::uint32_t bits = 0;
      for (std::uint32_t i = 0; i < words; ++i)
        bits += WvPopcount64(p[i]);
      bitsOut = bits;
      return true;
    }

    // THE REGION LAYOUT, RE-DERIVED FROM sub_1802E7D10 (the allocator).
    //
    // The previous version of this probe had it wrong in a way that made the
    // measurement un-interpretable, so the derivation is written out here in
    // full. sub_1802E7D10 writes every header field, so it is ground truth:
    //
    //   ctx[0x54070] = dword_181748DBC >> 6;                          // 126
    //   ctx[0x54074] = dword_181748DC0 >> 6;                          // 344
    //   ctx[0x54078] = ctx[0x54074] + sub_1801A8E80(&qword_180EAA020);// 344+128
    //   ctx[0x5407C] = ctx[0x54078] + ((dword_181748D8C + 63) >> 6);  // 472+126
    //   ctx[0x54080] = (dword_181748D8C + 63) >> 6;                   // 126
    //
    // Five bitmasks are packed back to back at +0x54088, in 64-bit words:
    //
    //   M1  [0,    c70)  world mask 1     — EF770 memsets [0,c74) to all-ones
    //   M2  [c70,  c74)  STATIC PROPS     — the slice handed to
    //                                       StaticPropMgrClient005 vtable+72
    //                                       (EF770) / +80 (EF510), as
    //                                       (base = c70, count = c74 - c70)
    //   R   [c74,  c78)  RENDERABLES      — what sub_1801A8350 BUILDS
    //   B   [c78,  c78+c80)  leaf/PVS input — EF510 memmoves this from its `a4`;
    //                                       EF770 fills it all-ones via
    //                                       sub_1805C4F60(base, c78<<6, D8C)
    //   T   [c7C,  c7C+N)    second renderable mask — memset 0xFF when
    //                                       *(a2+32) == 0
    //
    // where N = c78 - c74 = the renderable word count = *(a1+48).
    //
    // WHAT THE OLD PROBE ACTUALLY MEASURED, and why it could not resolve this:
    //   "A"    = base 0, len c78   -> M1 + M2 + R MERGED. A superset that
    //                                 CONTAINS the region it was being
    //                                 correlated against, so A-vs-prop tracking
    //                                 each other was partly tautological.
    //   "prop" = base c70, len 218 -> correct, that really is M2.
    //   "B"    = base c78, len c70 -> correct region, and correct length only by
    //                                 coincidence (c80 == c70 == 126 here).
    //
    // The consequence that matters: R was never isolated. It was buried inside
    // "A" under 344 words of world+prop bits. And the claim that sub_1801A8350
    // *reads* +0x54078 is inverted — at 0x1801A83D0 it takes [r8+54074h] as the
    // base of the mask it WRITES (v9/v71), and touches c78 only at 0x1801A84F8
    // as the base of the mask it READS. So the function is a B -> R transform:
    // B flat with pitch while R is unmeasured is exactly what a cull living
    // INSIDE this function looks like, not evidence against one.
    //
    // Hence: popcount all five regions separately, and read R after the call.
    struct WvSnap {
      bool valid = false;
      int hitArg = -1;
      std::uintptr_t ctx = 0;
      std::uint32_t c70 = 0, c74 = 0, c78 = 0, c7C = 0, c80 = 0;
      std::uint32_t nWords = 0;          // N = c78 - c74, the renderable count
      std::uint32_t m1 = 0, m2 = 0, r = 0, b = 0, t = 0;
      bool okM1 = false, okM2 = false, okR = false, okB = false, okT = false;
      bool layoutOk = false;             // c7C == c78 + c80
    };

    // Validate a candidate view-context pointer AND fill in the region map.
    //
    // The self-consistency check c7C == c78 + c80 is what makes this a real
    // identification rather than a range guess: it can only hold if the pointer
    // is a context laid out by sub_1802E7D10. It also verifies the derivation
    // above against the live process every single call.
    bool WvTryContext(std::uintptr_t ctx, WvSnap& s) {
      if (ctx == 0 || (ctx & 7) != 0)
        return false;
      if (!WvReadable(reinterpret_cast<void*>(ctx + kOffBitmaskHeader), 0x14))
        return false;
      const auto* h = reinterpret_cast<const std::uint32_t*>(ctx + kOffBitmaskHeader);
      const std::uint32_t c70 = h[0], c74 = h[1], c78 = h[2], c7C = h[3], c80 = h[4];
      // These are counts of 64-bit words. Anything outside this is not a context.
      if (c70 == 0 || c70 > 8192 || c74 > 8192 || c78 > 8192 || c7C > 8192 || c80 > 8192)
        return false;
      // Monotonic by construction — each field is the previous plus a length.
      if (!(c70 <= c74 && c74 <= c78 && c78 <= c7C))
        return false;
      std::uint32_t probe = 0;
      if (!WvPopRegion(ctx + kOffWorldBitmask, c70, probe))
        return false;
      s.ctx = ctx;
      s.c70 = c70; s.c74 = c74; s.c78 = c78; s.c7C = c7C; s.c80 = c80;
      s.nWords = c78 - c74;
      s.layoutOk = (c7C == c78 + c80);
      return true;
    }

    // Popcount all five regions of an already-validated context.
    void WvReadRegions(WvSnap& s) {
      const std::uintptr_t base = s.ctx + kOffWorldBitmask;
      s.okM1 = WvPopRegion(base,                    s.c70,          s.m1);
      s.okM2 = WvPopRegion(base + 8ull * s.c70,     s.c74 - s.c70,  s.m2);
      s.okR  = WvPopRegion(base + 8ull * s.c74,     s.nWords,       s.r);
      s.okB  = WvPopRegion(base + 8ull * s.c78,     s.c80,          s.b);
      s.okT  = WvPopRegion(base + 8ull * s.c7C,     s.nWords,       s.t);
    }

    // Read a ConVar reached through a GLOBAL POINTER (as opposed to a ConVar
    // object living at a fixed RVA). sub_1801A8350 and sub_1801A8E80 both do
    //     mov rax, cs:qword_...   /   cmp dword ptr [rax+5Ch], 0
    // so the value is *(int*)(*(void**)(client + rva) + 0x5C), one more
    // indirection than the [ClientProbe] table's fixed-object ConVars use.
    // Name comes from m_pszName at +0x18 and is logged once, because these two
    // globals have no other xref in the binary and cannot be named statically.
    bool WvReadConVarPtr(std::uintptr_t rva, std::int32_t& valOut, const char*& nameOut) {
      HMODULE cl = GetModuleHandleA("client.dll");
      if (cl == nullptr)
        return false;
      const auto slot = reinterpret_cast<std::uintptr_t>(cl) + rva;
      if (!WvReadable(reinterpret_cast<void*>(slot), sizeof(void*)))
        return false;
      const auto obj = *reinterpret_cast<const std::uintptr_t*>(slot);
      if (!WvReadable(reinterpret_cast<void*>(obj + 0x5C), sizeof(std::int32_t)))
        return false;
      valOut = *reinterpret_cast<const std::int32_t*>(obj + 0x5C);
      if (WvReadable(reinterpret_cast<void*>(obj + 0x18), sizeof(void*))) {
        const auto pName = *reinterpret_cast<const char* const*>(obj + 0x18);
        if (WvReadable(pName, 1))
          nameOut = pName;
      }
      return true;
    }

    // client.dll globals, IDA VA minus the 0x180000000 preferred base.
    //   0xEA9D78  qword_180EA9D78 — gates the "set a bit for EVERY occupied
    //             renderable slot" path at 0x1801A844E. Nonzero (and view flag
    //             0x100 clear) makes sub_1801A8350 skip the B->leaf->renderable
    //             expansion entirely. That is a shipping no-cull path.
    //   0x216FBF8 qword_18216FBF8 — gates sub_1801A8E80's 128-vs-*(a1+48) clamp,
    //             which is what sizes regions R and T.
    constexpr std::uintptr_t kRvaCvDrawAll   = 0xEA9D78;
    constexpr std::uintptr_t kRvaCvMaskClamp = 0x216FBF8;

    // ClientLeafSystem (arg 1 of sub_1801A8350, and the &qword_180EAA020 passed
    // to sub_1801A8E80) carries four parallel N-word masks that sub_1801A8350
    // combines into R. None of these have been measured before:
    //   +0x38   master "slot occupied" mask
    //   +0x438  force-visible mask, OR'd into R
    //   +0x838  AND mask applied on normal views      <-- wholesale cull
    //   +0xC38  AND mask applied when (a2+36) & 0xC   <-- wholesale cull
    // and *(a1+48) is N itself, which cross-checks c78-c74 from the context.
    constexpr std::uintptr_t kOffLeafWordCount = 0x30;
    constexpr std::uintptr_t kOffLeafMaster    = 0x38;
    constexpr std::uintptr_t kOffLeafForceVis  = 0x438;
    constexpr std::uintptr_t kOffLeafAndNormal = 0x838;
    constexpr std::uintptr_t kOffLeafAndShadow = 0xC38;

    void WorldVisPre(void* a1, void* a2, void* a3, void* a4, WvSnap& s) {
      // Try every arg as the context; report which validated. a3 is the known
      // answer, but the check is nearly free and it is what catches a build
      // whose signature differs.
      void* const args[4] = { a1, a2, a3, a4 };
      for (int i = 0; i < 4; ++i) {
        if (WvTryContext(reinterpret_cast<std::uintptr_t>(args[i]), s)) {
          s.hitArg = i;
          s.valid = true;
          break;
        }
      }
      if (!s.valid)
        return;
      WvReadRegions(s);
    }

    void WorldVisPost(void* a1, void* a2, void* /*a3*/, void* /*a4*/,
                      const WvSnap& pre, std::int64_t ret) {
      if (!pre.valid)
        return;

      // R and T are WRITTEN by the call, so they only mean anything afterwards.
      // M1/M2/B are inputs and are re-read only to prove the call left them
      // alone — if they move, the region model is still wrong.
      WvSnap post = pre;
      WvReadRegions(post);

      // Read pvs_debug EXACTLY as sub_1802ECFC0 reads it, so this cannot
      // disagree with the consumer:
      //     v5 = *(int*)(*(void**)(pvs_debug + 0x38) + 92)
      // i.e. deref m_pParent first, then m_nValue. [ClientProbe] writes
      // obj+0x5C DIRECTLY, which is only the same location while the ConVar is
      // its own parent (sub_180737E60 does `*(a1+56) = a1` at construction, so
      // it starts self-parented — but nothing guarantees it stays that way, and
      // "FORCE pvs_debug -> 2" logging at startup does NOT prove the value is
      // live at read time). Bit 1 set == the frustum walk is short-circuited.
      std::int32_t pvsDbg = -1;      // -1 = could not read
      if (HMODULE cl = GetModuleHandleA("client.dll")) {
        const auto obj = reinterpret_cast<std::uintptr_t>(cl) + 0x1748A00;
        if (WvReadable(reinterpret_cast<void*>(obj + 0x38), sizeof(void*))) {
          const auto parent = *reinterpret_cast<const std::uintptr_t*>(obj + 0x38);
          if (WvReadable(reinterpret_cast<void*>(parent + 92), sizeof(std::int32_t)))
            pvsDbg = *reinterpret_cast<const std::int32_t*>(parent + 92);
        }
      }

      // View flags — these select which of the two AND masks is applied, and
      // whether the shadow path (a2+32 == 2) runs instead of the frustum call.
      std::uint32_t vf20 = 0xFFFFFFFFu, vf24 = 0xFFFFFFFFu;
      const auto view = reinterpret_cast<std::uintptr_t>(a2);
      if (WvReadable(reinterpret_cast<void*>(view + 0x20), 8)) {
        vf20 = *reinterpret_cast<const std::uint32_t*>(view + 0x20);
        vf24 = *reinterpret_cast<const std::uint32_t*>(view + 0x24);
      }

      std::int32_t cvDrawAll = -1, cvClamp = -1;
      const char* cvDrawAllName = "?";
      const char* cvClampName = "?";
      const bool okCvDrawAll = WvReadConVarPtr(kRvaCvDrawAll, cvDrawAll, cvDrawAllName);
      WvReadConVarPtr(kRvaCvMaskClamp, cvClamp, cvClampName);

      // Which of the two builds ran, per the branch at 0x1801A842A/0x1801A844E:
      //   test [rsi+20h], 100h ; jnz expand
      //   cmp  [cvDrawAll+5Ch], 0 ; jz expand
      // "all" means it set a bit for every occupied renderable slot and never
      // touched region B. Reported as "?" rather than guessed if the ConVar
      // could not be read — a mislabelled path is worse than a missing one.
      const char* path = !okCvDrawAll ? "?"
                       : (((vf20 & 0x100) != 0 || cvDrawAll == 0) ? "expand" : "all");

      // ClientLeafSystem masks.
      const auto leaf = reinterpret_cast<std::uintptr_t>(a1);
      std::uint32_t lsN = 0, lsMaster = 0, lsForce = 0, lsAnd = 0, lsAnd2 = 0;
      bool leafOk = false;
      if (WvReadable(reinterpret_cast<void*>(leaf + kOffLeafWordCount), 4)) {
        lsN = *reinterpret_cast<const std::uint32_t*>(leaf + kOffLeafWordCount);
        if (lsN != 0 && lsN <= 8192) {
          leafOk = WvPopRegion(leaf + kOffLeafMaster,    lsN, lsMaster)
                && WvPopRegion(leaf + kOffLeafForceVis,  lsN, lsForce)
                && WvPopRegion(leaf + kOffLeafAndNormal, lsN, lsAnd)
                && WvPopRegion(leaf + kOffLeafAndShadow, lsN, lsAnd2);
        }
      }

      const std::uint64_t n = s_wvCalls.fetch_add(1, std::memory_order_relaxed);

      if (n == 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[WorldVis.Cv] drawAll(client+0x", std::hex, kRvaCvDrawAll, std::dec,
          ")=\"", cvDrawAllName, "\" val=", cvDrawAll,
          "  maskClamp(client+0x", std::hex, kRvaCvMaskClamp, std::dec,
          ")=\"", cvClampName, "\" val=", cvClamp));
      }

      // Bit counts every call for the first 200, then 1-in-16 so the series
      // stays dense enough to correlate against per-frame [PitchProbe] lines
      // without dominating the log.
      if (n < 200 || (n & 15) == 0) {
        // rPost is the number one: it is the renderable visibility population
        // that this call actually produced. rPre should read ~0 (EF770/EF510
        // zero R before the job runs); if it does not, something else writes R
        // first and that is a finding in itself.
        dxvk::Logger::warn(dxvk::str::format(
          "[WorldVis] n=", n,
          " pvsDbg=", pvsDbg,
          " ctxArg=a", (pre.hitArg + 1),
          " layoutOk=", (pre.layoutOk ? 1 : 0),
          " c70=", pre.c70, " c74=", pre.c74, " c78=", pre.c78,
          " c7C=", pre.c7C, " c80=", pre.c80, " N=", pre.nWords,
          " m1=", (post.okM1 ? static_cast<std::int64_t>(post.m1) : -1),
          " m2=", (post.okM2 ? static_cast<std::int64_t>(post.m2) : -1),
          " b=", (post.okB ? static_cast<std::int64_t>(post.b) : -1),
          " rPre=", (pre.okR ? static_cast<std::int64_t>(pre.r) : -1),
          " rPost=", (post.okR ? static_cast<std::int64_t>(post.r) : -1),
          " tPost=", (post.okT ? static_cast<std::int64_t>(post.t) : -1),
          " ret=", ret,
          " path=", path,
          " vf20=0x", std::hex, vf20, " vf24=0x", vf24, std::dec,
          " ctx=0x", std::hex, pre.ctx, std::dec));

        dxvk::Logger::warn(dxvk::str::format(
          "[WorldVis.Leaf] n=", n,
          " ok=", (leafOk ? 1 : 0),
          " lsN=", lsN, " (ctxN=", pre.nWords, ")",
          " master=", lsMaster,
          " forceVis=", lsForce,
          " andNormal=", lsAnd,
          " andShadow=", lsAnd2));
      }

      // Caller chain, only for the first few calls — this is what names the
      // producer. Frames are printed as module+RVA so they paste straight into
      // IDA. Skip 1 frame to drop WorldVisPost itself.
      if (n < 6) {
        void* frames[16] = {};
        const USHORT captured = RtlCaptureStackBackTrace(1, 16, frames, nullptr);
        // Modules are tested against their real SizeOfImage, so an address just
        // past one module can no longer be reported as a huge offset into it.
        static const char* const kMods[] = {
          "engine.dll", "client.dll", "d3d11.dll", "materialsystem_dx11.dll",
          "studiorender.dll", "vgui2.dll",
        };
        for (USHORT i = 0; i < captured; ++i) {
          const auto addr = reinterpret_cast<std::uintptr_t>(frames[i]);
          std::string where;
          std::uintptr_t rva = 0;
          for (const char* mod : kMods) {
            if (WvModuleRange(mod, addr, rva)) {
              where = dxvk::str::format(mod, "+0x", std::hex, rva, std::dec);
              break;
            }
          }
          if (where.empty())
            where = dxvk::str::format("0x", std::hex, addr, std::dec);
          dxvk::Logger::warn(dxvk::str::format(
            "[WorldVis.Stack] n=", n, " frame[", i, "]=", where));
        }
      }
    }

    std::int64_t __fastcall WorldVisWrapper(void* a1, void* a2, void* a3, void* a4) {
      // R is written BY the call, so the interesting read has to happen after
      // it returns. Snapshot the inputs first, run the original, then report.
      // WvSnap is trivially destructible, which is what lets it coexist with
      // __try in the same function (MSVC C2712 otherwise).
      WvSnap snap;
      __try {
        WorldVisPre(a1, a2, a3, a4, snap);
      } __except (EXCEPTION_EXECUTE_HANDLER) {
        snap.valid = false;
      }

      // Never let a diagnostic take down the game — this runs on the render
      // path and a fault here would be indistinguishable from the bug.
      const std::int64_t ret = s_origWorldVis(a1, a2, a3, a4);

      __try {
        WorldVisPost(a1, a2, a3, a4, snap, ret);
      } __except (EXCEPTION_EXECUTE_HANDLER) {
      }
      return ret;
    }

    bool DoInstallWorldVis() {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr) {
        dxvk::Logger::warn("[WorldVis] client.dll not loaded yet");
        return false;
      }
      auto* target = reinterpret_cast<std::uint8_t*>(client) + kRvaWorldVisConsumer;

      // Verify the prologue before touching anything. Wrong build => no patch.
      static constexpr std::uint8_t kExpect[kWvPrologueSize] = {
        0x4C, 0x89, 0x4C, 0x24, 0x20,  // mov [rsp+arg_18], r9
      };
      if (std::memcmp(target, kExpect, kWvPrologueSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[WorldVis] prologue mismatch at client.dll+0x", std::hex,
          kRvaWorldVisConsumer, std::dec, " — not this build, hook skipped"));
        return false;
      }

      // TWO HOPS ARE REQUIRED, and the first attempt got this wrong.
      //
      // Only 9 bytes can be stolen (see kWvPrologueSize), which fits a 5-byte
      // `E9 rel32` and nothing larger. But the wrapper lives in d3d11.dll,
      // which Windows maps arbitrarily far from the hooked module — far beyond
      // the +/-2GB a rel32 can reach. Allocating the trampoline near the target
      // does nothing for the OUTBOUND jump; the patch site still has to reach
      // the wrapper. So the near page carries a GATEWAY: the patch site does a
      // rel32 jmp to the gateway (near, always reachable), and the gateway does
      // a 14-byte absolute jmp to the wrapper (anywhere in the address space).
      //
      // Near page layout:
      //   +0   gateway:    FF 25 00000000 <abs64 wrapper>       14 bytes
      //   +16  trampoline: <stolen 9 bytes>
      //   +25              FF 25 00000000 <abs64 target+9>      14 bytes
      std::uint8_t* page = AllocateNearPage(reinterpret_cast<std::uint8_t*>(client));
      if (page == nullptr) {
        dxvk::Logger::warn("[WorldVis] no page within rel32 range of client.dll");
        return false;
      }

      const auto wrapperAddr = reinterpret_cast<std::uintptr_t>(&WorldVisWrapper);

      // Gateway at +0: absolute indirect jmp to the wrapper.
      std::uint8_t* gate = page;
      gate[0] = 0xFF; gate[1] = 0x25;
      gate[2] = 0x00; gate[3] = 0x00; gate[4] = 0x00; gate[5] = 0x00;
      const auto absWrapper = static_cast<std::uint64_t>(wrapperAddr);
      std::memcpy(gate + 6, &absWrapper, sizeof(absWrapper));

      // Trampoline at +16: stolen prologue, then absolute jmp back past it.
      std::uint8_t* tramp = page + 16;
      std::memcpy(tramp, target, kWvPrologueSize);
      std::uint8_t* jb = tramp + kWvPrologueSize;
      jb[0] = 0xFF; jb[1] = 0x25; jb[2] = 0x00; jb[3] = 0x00; jb[4] = 0x00; jb[5] = 0x00;
      const auto retAddr = reinterpret_cast<std::uint64_t>(target + kWvPrologueSize);
      std::memcpy(jb + 6, &retAddr, sizeof(retAddr));
      s_origWorldVis = reinterpret_cast<WorldVisFn>(tramp);

      // Patch site: 5-byte rel32 jmp to the GATEWAY (not the wrapper). The
      // stolen prologue is exactly 5 bytes, so no padding nops are needed.
      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(gate)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[WorldVis] gateway out of rel32 range (rel=", rel, "), hook skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, kWvPrologueSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn("[WorldVis] VirtualProtect failed");
        return false;
      }
      std::uint8_t stub[kWvPrologueSize] = { 0xE9, 0, 0, 0, 0 };
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, kWvPrologueSize);
      DWORD tmp = 0;
      VirtualProtect(target, kWvPrologueSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, kWvPrologueSize);

      dxvk::Logger::warn(dxvk::str::format(
        "[WorldVis] INSTALLED at client.dll+0x", std::hex, kRvaWorldVisConsumer,
        " target=0x", reinterpret_cast<std::uintptr_t>(target),
        " gateway=0x", reinterpret_cast<std::uintptr_t>(gate),
        " trampoline=0x", reinterpret_cast<std::uintptr_t>(tramp),
        " wrapper=0x", wrapperAddr, std::dec));
      return true;
    }

    std::once_flag s_onceWorldVis;
    std::atomic<bool> s_worldVisInstalled{false};

    // =======================================================================
    // NV-DXVK [JobProbe] — how many JOBS is the world visibility worker given?
    //
    // WHAT THIS REPLACES. HANDOFF_PITCH_CULL_2026-08-05 §10.1 prescribes an
    // x64dbg zero-pause hit counter on the node loop head (client+0x2E8F50)
    // as the next move. That measurement was taken and it works, but it costs
    // one debug exception per hit: ~1600 hits/frame took the game to 0.5 fps
    // and crashed it twice. This is the same measurement in-process.
    //
    // WHY NOT COUNT NODES. A debugger run with two counters (worker entry and
    // node loop head) measured 6.46 nodes/call at pitch 5.8 deg vs 9.98 at
    // 89 deg — node iterations go UP looking down. That is not a surprise
    // once the loop is read properly: var_12A0 is popped from a 1024-entry
    // ring queue the function fills itself (see the header comment), so with
    // the §2 reject branches NOPed the traversal expands rather than prunes.
    // §6's premise — "the loop bounds come from the caller" — is false, and
    // its proposed test ("node iterations drop ~25% => node supply is the
    // residual") cannot discriminate anything.
    //
    // WHAT THIS PROBE THEN MEASURED, 2026-08-05, over 429 frames on a camera
    // that moved 9 units, binned in 10-degree steps to 80 deg:
    //
    //   calls/frame  192 206 207 186 186 184 184 184   (sd=0 on the last three)
    //   inst         644 621 608 557 511 498 555 581   (r = -0.77)
    //
    // Job supply is FLAT — 4% and non-monotonic — while instances fall 23%.
    // recCntSum per frame RISES with pitch (769 -> 877), so it is not a
    // disguised reduction either. sub_1802EB620 is therefore excluded BY
    // MEASUREMENT, not by argument.
    //
    // NOTE: an earlier reading of this, from two CROSS-SESSION debugger
    // samples, put calls/frame at ~321 forward vs ~250 down and concluded the
    // job count carried the view dependence. That was wrong — the breakpoint
    // overhead and the session/position difference produced it, not pitch. It
    // is recorded here only so the number is not resurrected from the older
    // notes. The open question moved to the worker's OUTPUT; see [DrainProbe].
    //
    // HOW THE JOB IS DECODED, from the prologue (all displacements are
    // image-base relative; rsi = 0x180000000 in the listing):
    //
    //   mov   rbx, rdx                       ; a2 = job index
    //   lea   r11, ds:13C0980h[rbx*4]        ; r11 = jobArray + a2*4
    //   add   r11, rsi
    //   movzx eax, word ptr [r11]            ; w0
    //   shl   rax, 6                         ; w0 * 64
    //   add   rax, rcx                       ; rcx = unk_181380A40
    //   movzx r12d, word ptr [rax]           ; recCnt  <-- sizes the dword
    //   lea   r10, [rax+8]                   ;             array at rax+8
    //   movzx eax, word ptr [r11+2]          ; w1 = seed BSP node index
    //
    // recCnt is logged as a supply proxy. Its exact semantics are NOT
    // confirmed — it is the count word that sizes the per-record dword array,
    // nothing more is claimed for it. `calls` is the number that matters.
    //
    // Cost: a handful of loads and four relaxed atomic adds per job call. No
    // allocation, no lock, no logging on the job threads at all — the frame
    // loop drains the counters and emits exactly one line.
    // =======================================================================
    using WorldJobFn = std::int64_t(__fastcall*)(void*, std::uint64_t, void*, void*);
    WorldJobFn s_origWorldJob = nullptr;

    constexpr std::uintptr_t kRvaWorldJobWorker = 0x2E8DA0;   // sub_1802E8DA0
    constexpr std::uintptr_t kRvaJobArray       = 0x13C0980;  // job entries, 4 bytes each
    constexpr std::uintptr_t kRvaJobRecTable    = 0x1380A40;  // unk_181380A40, 64-byte stride

    // sub_1802E8DA0 prologue:
    //   +0  48 89 5C 24 08   mov [rsp+arg_0], rbx   <-- 5 bytes, exactly a jmp
    //   +5  48 89 74 24 10   mov [rsp+arg_8], rsi
    // The first instruction is 5 bytes and position independent, so it is
    // stolen whole and nothing rip-relative is relocated. Same two-hop
    // gateway scheme as [WorldVis] — see DoInstallWorldVis for why the
    // outbound jump cannot reach d3d11.dll directly.
    constexpr std::size_t kWjPrologueSize = 5;

    // Atomic, and published BEFORE the patch site is written: the moment the
    // jmp lands, job threads are already inside the wrapper, so a plain
    // pointer stored afterwards would be a genuine (if narrow) data race.
    std::atomic<std::uint8_t*> s_wjClientBase{nullptr};

    // The accumulators live in the DXVK layer (dxvk::tf2, defined in
    // rtx_render/rtx_camera_manager.cpp) rather than in this file. The reader
    // is rtx_instance_manager.cpp, which compiles into libdxvk.a — and
    // dxgi.dll links libdxvk.a with NO d3d11 objects, so a call from there
    // into this namespace links for d3d11.dll and fails dxgi.dll with
    // LNK2019. Pushing the state down one layer makes the dependency point
    // the way the link does. Same arrangement as g_engineHookCaptureCount.
    inline void WjAtomicMin(std::atomic<std::uint32_t>& slot, std::uint32_t v) {
      std::uint32_t cur = slot.load(std::memory_order_relaxed);
      while (v < cur && !slot.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
      }
    }

    inline void WjAtomicMax(std::atomic<std::uint32_t>& slot, std::uint32_t v) {
      std::uint32_t cur = slot.load(std::memory_order_relaxed);
      while (v > cur && !slot.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
      }
    }

    // Runs on the job threads. Must not log, allocate, or fault.
    void WorldJobSample(std::uint64_t jobIdx) {
      dxvk::tf2::g_jobProbeCalls.fetch_add(1, std::memory_order_relaxed);

      const std::uint8_t* client = s_wjClientBase.load(std::memory_order_acquire);
      if (client == nullptr) {
        dxvk::tf2::g_jobProbeBadReads.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      // The index is scaled by 4 into a global array with no bound available
      // to us, so clamp to something no sane job count can exceed before
      // touching memory. 0x10000 entries = 256KB, well past the ~few-hundred
      // jobs observed.
      if (jobIdx >= 0x10000ull) {
        dxvk::tf2::g_jobProbeBadReads.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      const auto idx32 = static_cast<std::uint32_t>(jobIdx);
      WjAtomicMin(dxvk::tf2::g_jobProbeJobIdxMin, idx32);
      WjAtomicMax(dxvk::tf2::g_jobProbeJobIdxMax, idx32);

      // dword_1813C0940 was sampled here originally, on the theory that it
      // held the job count. Measured, it does not: it churns randomly over
      // 2.4M-4.5M with the next three dwords always 0, and tracks nothing.
      // Removed rather than left in — it cost a VirtualQuery per job call
      // (~200/frame) to log a number with no meaning.
      const auto* entry = reinterpret_cast<const std::uint16_t*>(
        client + kRvaJobArray + jobIdx * 4);
      if (!WvReadable(entry, 4)) {
        dxvk::tf2::g_jobProbeBadReads.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      const std::uint16_t w0 = entry[0];
      const auto* rec = reinterpret_cast<const std::uint16_t*>(
        client + kRvaJobRecTable + (static_cast<std::size_t>(w0) << 6));
      if (!WvReadable(rec, 2)) {
        dxvk::tf2::g_jobProbeBadReads.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      dxvk::tf2::g_jobProbeRecCntSum.fetch_add(rec[0], std::memory_order_relaxed);
    }

    std::int64_t __fastcall WorldJobWrapper(void* a1, std::uint64_t a2, void* a3, void* a4) {
      // Sample before the call: the job index is an input, and reading it
      // after would race the worker's own writes. A diagnostic must never be
      // the thing that takes the game down, hence the SEH wrap — this runs on
      // the job threads and a fault here would look exactly like the bug.
      __try {
        WorldJobSample(a2);
      } __except (EXCEPTION_EXECUTE_HANDLER) {
        dxvk::tf2::g_jobProbeBadReads.fetch_add(1, std::memory_order_relaxed);
      }
      return s_origWorldJob(a1, a2, a3, a4);
    }

    bool DoInstallWorldJob() {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr) {
        dxvk::Logger::warn("[JobProbe] client.dll not loaded yet");
        return false;
      }
      auto* target = reinterpret_cast<std::uint8_t*>(client) + kRvaWorldJobWorker;

      static constexpr std::uint8_t kExpect[kWjPrologueSize] = {
        0x48, 0x89, 0x5C, 0x24, 0x08,  // mov [rsp+arg_0], rbx
      };
      if (std::memcmp(target, kExpect, kWjPrologueSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[JobProbe] prologue mismatch at client.dll+0x", std::hex,
          kRvaWorldJobWorker, std::dec, " — not this build, hook skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(reinterpret_cast<std::uint8_t*>(client));
      if (page == nullptr) {
        dxvk::Logger::warn("[JobProbe] no page within rel32 range of client.dll");
        return false;
      }

      const auto wrapperAddr = reinterpret_cast<std::uintptr_t>(&WorldJobWrapper);

      // Gateway at +0: absolute indirect jmp to the wrapper.
      std::uint8_t* gate = page;
      gate[0] = 0xFF; gate[1] = 0x25;
      gate[2] = 0x00; gate[3] = 0x00; gate[4] = 0x00; gate[5] = 0x00;
      const auto absWrapper = static_cast<std::uint64_t>(wrapperAddr);
      std::memcpy(gate + 6, &absWrapper, sizeof(absWrapper));

      // Trampoline at +16: stolen prologue, then absolute jmp back past it.
      std::uint8_t* tramp = page + 16;
      std::memcpy(tramp, target, kWjPrologueSize);
      std::uint8_t* jb = tramp + kWjPrologueSize;
      jb[0] = 0xFF; jb[1] = 0x25; jb[2] = 0x00; jb[3] = 0x00; jb[4] = 0x00; jb[5] = 0x00;
      const auto retAddr = reinterpret_cast<std::uint64_t>(target + kWjPrologueSize);
      std::memcpy(jb + 6, &retAddr, sizeof(retAddr));
      s_origWorldJob = reinterpret_cast<WorldJobFn>(tramp);

      // Published before the patch — see the declaration.
      s_wjClientBase.store(reinterpret_cast<std::uint8_t*>(client), std::memory_order_release);

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(gate)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[JobProbe] gateway out of rel32 range (rel=", rel, "), hook skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, kWjPrologueSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn("[JobProbe] VirtualProtect failed");
        return false;
      }
      std::uint8_t stub[kWjPrologueSize] = { 0xE9, 0, 0, 0, 0 };
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, kWjPrologueSize);
      DWORD tmp = 0;
      VirtualProtect(target, kWjPrologueSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, kWjPrologueSize);

      dxvk::Logger::warn(dxvk::str::format(
        "[JobProbe] INSTALLED at client.dll+0x", std::hex, kRvaWorldJobWorker,
        " target=0x", reinterpret_cast<std::uintptr_t>(target),
        " gateway=0x", reinterpret_cast<std::uintptr_t>(gate),
        " trampoline=0x", reinterpret_cast<std::uintptr_t>(tramp),
        " wrapper=0x", wrapperAddr, std::dec));
      return true;
    }

    std::once_flag s_onceWorldJob;
    std::atomic<bool> s_worldJobInstalled{false};

    // =======================================================================
    // NV-DXVK [DrainProbe] — the world visibility worker's OUTPUT.
    //
    // WHY THIS EXISTS. [JobProbe] closed the input question: job supply is
    // flat across pitch (192 -> 184 calls/frame, non-monotonic, sd=0 over the
    // top three bins, recCntSum RISING with pitch) while instances fall 23%
    // (r = -0.77) on a camera that moved 9 units. Handoff §6's proposed dig
    // into sub_1802EB620 is therefore excluded BY MEASUREMENT, not argument.
    //
    // The remaining fork is whether the mask the worker produces is full or
    // short. sub_1802F04F0 is a pure OR-in of accepted leaf runs — verified
    // exhaustively over its four nested loops, the only store is
    //
    //   1802f0573  mov   rdx, cs:qword_1811FC0C8   ; mask base, a GLOBAL
    //   1802f0588  lea   r9, [rdx+r8*8]
    //   1802f0594  or    [r9], rdx                 ; no compare on any path
    //
    // so a popcount of that mask after it returns IS the accepted-leaf count.
    //
    // HOW THE REGIONS ARE FOUND WITHOUT THE CONTEXT POINTER. The drain never
    // takes ctx — it reaches the mask through the global qword_1811FC0C8,
    // which is exactly why the +0x54088 displacement scan never found this
    // writer (handoff §8). But that global IS ctx+0x54088, so the five-DWORD
    // region header at ctx+0x54070 sits at maskBase - 0x18. That is derived,
    // not guessed, and it is checked: c70 < c74 < c78 and c7C == c78 + c80,
    // the same structural test [WorldVis] uses. layoutOk=0 => do not
    // interpret the run.
    //
    // The earlier [WorldVis] trap is deliberately not repeated here: this
    // reads the mask on the way OUT, because the call is what fills it.
    //
    // Cost: one popcount pass over c78 (~472) words per drain call.
    // =======================================================================
    using WorldDrainFn = std::int64_t(__fastcall*)(void*);
    WorldDrainFn s_origWorldDrain = nullptr;

    constexpr std::uintptr_t kRvaWorldDrain     = 0x2F04F0;    // sub_1802F04F0
    constexpr std::uintptr_t kRvaDrainMaskBase  = 0x11FC0C8;   // qword_1811FC0C8
    constexpr std::uintptr_t kOffDrainHdrFromMask = 0x18;      // ctx+0x54070 vs +0x54088

    // sub_1802F04F0 prologue:
    //   +0  48 89 5C 24 08   mov [rsp+arg_0], rbx   <-- 5 bytes, exactly a jmp
    // Identical shape to sub_1802E8DA0's, and equally position independent.
    constexpr std::size_t kWdPrologueSize = 5;

    std::atomic<std::uint8_t*> s_wdClientBase{nullptr};

    // Runs after the drain. Must not log, allocate, or fault.
    void WorldDrainSample() {
      dxvk::tf2::g_drainProbeCalls.fetch_add(1, std::memory_order_relaxed);

      const std::uint8_t* client = s_wdClientBase.load(std::memory_order_acquire);
      if (client == nullptr) {
        dxvk::tf2::g_drainProbeBad.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      const auto* pMask = reinterpret_cast<const std::uint8_t* const*>(
        client + kRvaDrainMaskBase);
      if (!WvReadable(pMask, sizeof(void*))) {
        dxvk::tf2::g_drainProbeBad.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      const std::uint8_t* mask = *pMask;
      if (mask == nullptr) {
        dxvk::tf2::g_drainProbeBad.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      const auto* hdr = reinterpret_cast<const std::uint32_t*>(
        mask - kOffDrainHdrFromMask);
      if (!WvReadable(hdr, sizeof(std::uint32_t) * 5)) {
        dxvk::tf2::g_drainProbeBad.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      const std::uint32_t c70 = hdr[0], c74 = hdr[1], c78 = hdr[2];
      const std::uint32_t c7C = hdr[3], c80 = hdr[4];

      // Structural validation, same test as [WorldVis]: only a context laid
      // out by sub_1802E7D10 satisfies this. Without it a stale or unrelated
      // pointer would popcount arbitrary memory and read as a plausible
      // number, which is the failure mode that cost this investigation two
      // earlier rounds.
      const bool layoutOk = (c70 < c74) && (c74 < c78)
                         && (c7C == c78 + c80) && (c78 <= 8192);
      if (!layoutOk) {
        dxvk::tf2::g_drainProbeLayoutOk.store(0, std::memory_order_relaxed);
        dxvk::tf2::g_drainProbeBad.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      const auto maskAddr = reinterpret_cast<std::uintptr_t>(mask);
      std::uint32_t m1 = 0, m2 = 0, r = 0;
      const bool ok = WvPopRegion(maskAddr, c70, m1)
                   && WvPopRegion(maskAddr + 8ull * c70, c74 - c70, m2)
                   && WvPopRegion(maskAddr + 8ull * c74, c78 - c74, r);
      if (!ok) {
        dxvk::tf2::g_drainProbeBad.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      dxvk::tf2::g_drainProbeLayoutOk.store(1, std::memory_order_relaxed);
      dxvk::tf2::g_drainProbeC70.store(c70, std::memory_order_relaxed);
      dxvk::tf2::g_drainProbeC74.store(c74, std::memory_order_relaxed);
      dxvk::tf2::g_drainProbeC78.store(c78, std::memory_order_relaxed);

      dxvk::tf2::g_drainProbeM1Sum.fetch_add(m1, std::memory_order_relaxed);
      dxvk::tf2::g_drainProbeM2Sum.fetch_add(m2, std::memory_order_relaxed);
      dxvk::tf2::g_drainProbeRSum.fetch_add(r, std::memory_order_relaxed);
      WjAtomicMax(dxvk::tf2::g_drainProbeM1Max, m1);
      WjAtomicMax(dxvk::tf2::g_drainProbeM2Max, m2);
    }

    std::int64_t __fastcall WorldDrainWrapper(void* a1) {
      // The call is what FILLS the mask, so the read has to happen after it
      // returns — reading at entry would give the memset-to-zero state, which
      // is precisely the trap the first [WorldVis] fell into.
      const std::int64_t ret = s_origWorldDrain(a1);
      __try {
        WorldDrainSample();
      } __except (EXCEPTION_EXECUTE_HANDLER) {
        dxvk::tf2::g_drainProbeBad.fetch_add(1, std::memory_order_relaxed);
      }
      return ret;
    }

    bool DoInstallWorldDrain() {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr) {
        dxvk::Logger::warn("[DrainProbe] client.dll not loaded yet");
        return false;
      }
      auto* target = reinterpret_cast<std::uint8_t*>(client) + kRvaWorldDrain;

      static constexpr std::uint8_t kExpect[kWdPrologueSize] = {
        0x48, 0x89, 0x5C, 0x24, 0x08,  // mov [rsp+arg_0], rbx
      };
      if (std::memcmp(target, kExpect, kWdPrologueSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[DrainProbe] prologue mismatch at client.dll+0x", std::hex,
          kRvaWorldDrain, std::dec, " — not this build, hook skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(reinterpret_cast<std::uint8_t*>(client));
      if (page == nullptr) {
        dxvk::Logger::warn("[DrainProbe] no page within rel32 range of client.dll");
        return false;
      }

      const auto wrapperAddr = reinterpret_cast<std::uintptr_t>(&WorldDrainWrapper);

      std::uint8_t* gate = page;
      gate[0] = 0xFF; gate[1] = 0x25;
      gate[2] = 0x00; gate[3] = 0x00; gate[4] = 0x00; gate[5] = 0x00;
      const auto absWrapper = static_cast<std::uint64_t>(wrapperAddr);
      std::memcpy(gate + 6, &absWrapper, sizeof(absWrapper));

      std::uint8_t* tramp = page + 16;
      std::memcpy(tramp, target, kWdPrologueSize);
      std::uint8_t* jb = tramp + kWdPrologueSize;
      jb[0] = 0xFF; jb[1] = 0x25; jb[2] = 0x00; jb[3] = 0x00; jb[4] = 0x00; jb[5] = 0x00;
      const auto retAddr = reinterpret_cast<std::uint64_t>(target + kWdPrologueSize);
      std::memcpy(jb + 6, &retAddr, sizeof(retAddr));
      s_origWorldDrain = reinterpret_cast<WorldDrainFn>(tramp);

      s_wdClientBase.store(reinterpret_cast<std::uint8_t*>(client), std::memory_order_release);

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(gate)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[DrainProbe] gateway out of rel32 range (rel=", rel, "), hook skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, kWdPrologueSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn("[DrainProbe] VirtualProtect failed");
        return false;
      }
      std::uint8_t stub[kWdPrologueSize] = { 0xE9, 0, 0, 0, 0 };
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, kWdPrologueSize);
      DWORD tmp = 0;
      VirtualProtect(target, kWdPrologueSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, kWdPrologueSize);

      dxvk::Logger::warn(dxvk::str::format(
        "[DrainProbe] INSTALLED at client.dll+0x", std::hex, kRvaWorldDrain,
        " target=0x", reinterpret_cast<std::uintptr_t>(target),
        " gateway=0x", reinterpret_cast<std::uintptr_t>(gate),
        " trampoline=0x", reinterpret_cast<std::uintptr_t>(tramp),
        " wrapper=0x", wrapperAddr, std::dec));
      return true;
    }

    std::once_flag s_onceWorldDrain;
    std::atomic<bool> s_worldDrainInstalled{false};
  }  // namespace

  bool EnsureWorldVisHookInstalled() {
    std::call_once(s_onceWorldVis, []() {
      s_worldVisInstalled.store(DoInstallWorldVis(), std::memory_order_release);
    });
    return s_worldVisInstalled.load(std::memory_order_acquire);
  }

  bool EnsureWorldJobHookInstalled() {
    std::call_once(s_onceWorldJob, []() {
      s_worldJobInstalled.store(DoInstallWorldJob(), std::memory_order_release);
    });
    return s_worldJobInstalled.load(std::memory_order_acquire);
  }

  bool EnsureWorldDrainHookInstalled() {
    std::call_once(s_onceWorldDrain, []() {
      s_worldDrainInstalled.store(DoInstallWorldDrain(), std::memory_order_release);
    });
    return s_worldDrainInstalled.load(std::memory_order_acquire);
  }

}  // namespace tf2_decal_hook

// ===========================================================================
// NV-DXVK [tf2_engine_cvars]
// ===========================================================================
namespace tf2_engine_cvars {

  namespace {
    // ConVar field offsets — verified against engine.dll's ConVar ctor
    // (sub_180416A40), which stores: [a1+56] = a1 (m_pParent self-pointer),
    // [a1+88] = atof(default) (m_fValue), [a1+92] = (int)m_fValue (m_nValue),
    // [a1+24] = name pointer, [a1+40] = flags.
    constexpr std::size_t kOffName   = 0x18;
    constexpr std::size_t kOffParent = 0x38;
    constexpr std::size_t kOffFValue = 0x58;
    constexpr std::size_t kOffNValue = 0x5C;

    // RVAs of the ConVar OBJECTS (not the m_pParent slots) in engine.dll.
    // Taken from the static ConVar constructors:
    //   sub_1805B0380 -> unk_193EC7900  staticProp_earlyDepthPrepass
    //   sub_1805B03C0 -> unk_193B87470  staticProp_earlyDepthPrepassDist
    //   sub_1805B0400 -> unk_193B87350  ...IncludeOpaques
    //   sub_1805B0440 -> unk_193B87110  ...IncludeOpaquesDist
    //   sub_1805B0340 -> unk_193EC7870  staticProp_drawDecalsInSortOrder
    // minus the 0x180000000 preferred base.
    struct CvarDef {
      const char*   name;
      std::uintptr_t rva;
    };

    enum CvarIndex : int {
      kEarlyDepthPrepass = 0,
      kEarlyDepthPrepassDist,
      kIncludeOpaques,
      kIncludeOpaquesDist,
      kDrawDecalsInSortOrder,
      kCvarCount
    };

    constexpr CvarDef kCvars[kCvarCount] = {
      { "staticProp_earlyDepthPrepass",                   0x13EC7900 },
      { "staticProp_earlyDepthPrepassDist",               0x13B87470 },
      { "staticProp_earlyDepthPrepassIncludeOpaques",     0x13B87350 },
      { "staticProp_earlyDepthPrepassIncludeOpaquesDist", 0x13B87110 },
      { "staticProp_drawDecalsInSortOrder",               0x13EC7870 },
    };

    // True iff [p, p+len) is committed and readable.
    bool IsReadable(const void* p, std::size_t len) {
      if (p == nullptr)
        return false;
      MEMORY_BASIC_INFORMATION mbi = {};
      if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0)
        return false;
      if (mbi.State != MEM_COMMIT)
        return false;
      constexpr DWORD kNoAccess = PAGE_NOACCESS | PAGE_GUARD;
      if ((mbi.Protect & kNoAccess) != 0)
        return false;
      const auto start = reinterpret_cast<std::uintptr_t>(p);
      const auto regionEnd =
        reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
      return start + len <= regionEnd;
    }

    // Resolve a cvar object and confirm its identity by name. Returns the
    // address to write (the parent), or nullptr if anything looks wrong.
    std::uint8_t* ResolveAndVerify(int index) {
      // Never cache a foreign module's base — re-resolve every call. The
      // engine can unload/reload DLLs and a stale base is an AV.
      HMODULE engine = GetModuleHandleA("engine.dll");
      if (engine == nullptr)
        return nullptr;

      auto* obj = reinterpret_cast<std::uint8_t*>(engine) + kCvars[index].rva;
      if (!IsReadable(obj, kOffNValue + sizeof(std::int32_t)))
        return nullptr;

      const char* name = *reinterpret_cast<const char* const*>(obj + kOffName);
      if (!IsReadable(name, 1))
        return nullptr;
      if (std::strcmp(name, kCvars[index].name) != 0)
        return nullptr;

      auto* parent = *reinterpret_cast<std::uint8_t* const*>(obj + kOffParent);
      if (parent == nullptr)
        parent = obj;  // pre-registration, or a build without the self-link
      if (!IsReadable(parent, kOffNValue + sizeof(std::int32_t)))
        return nullptr;
      return parent;
    }

    // Sentinel: rtx.conf value < 0 means "leave the engine default alone".
    constexpr float kLeaveAlone = -1.0f;

    // Remember what we last wrote so the per-frame call only logs on change.
    float s_lastApplied[kCvarCount] = {
      kLeaveAlone, kLeaveAlone, kLeaveAlone, kLeaveAlone, kLeaveAlone
    };
    bool s_warnedUnresolved[kCvarCount] = { false, false, false, false, false };

    void ApplyOne(int index, float value) {
      if (value < 0.0f)
        return;  // sentinel — not overridden

      std::uint8_t* parent = ResolveAndVerify(index);
      if (parent == nullptr) {
        if (!s_warnedUnresolved[index]) {
          s_warnedUnresolved[index] = true;
          dxvk::Logger::warn(dxvk::str::format(
            "[tf2_engine_cvars] could NOT verify '", kCvars[index].name,
            "' at engine.dll+0x", std::hex, kCvars[index].rva, std::dec,
            " — name check failed or memory unreadable. Skipping the write "
            "(fail-closed). Probable cause: different engine.dll build."));
        }
        return;
      }

      // .data is already RW in a loaded PE image, so no VirtualProtect.
      *reinterpret_cast<float*>(parent + kOffFValue) = value;
      *reinterpret_cast<std::int32_t*>(parent + kOffNValue) =
        static_cast<std::int32_t>(value);

      if (s_lastApplied[index] != value) {
        s_lastApplied[index] = value;
        dxvk::Logger::info(dxvk::str::format(
          "[tf2_engine_cvars] ", kCvars[index].name, " = ", value));
      }
    }
  }  // namespace

  void ApplyOverrides(const Overrides& overrides) {
    if (!overrides.enabled)
      return;

    ApplyOne(kEarlyDepthPrepass,     overrides.earlyDepthPrepass);
    ApplyOne(kEarlyDepthPrepassDist, overrides.earlyDepthPrepassDist);
    ApplyOne(kIncludeOpaques,        overrides.includeOpaques);
    ApplyOne(kIncludeOpaquesDist,    overrides.includeOpaquesDist);
    ApplyOne(kDrawDecalsInSortOrder, overrides.drawDecalsInSortOrder);
  }

}  // namespace tf2_engine_cvars
