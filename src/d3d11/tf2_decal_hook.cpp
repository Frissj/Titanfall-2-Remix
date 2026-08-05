#include "tf2_decal_hook.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <windows.h>
// _ReturnAddress() — used to attribute sub_1802E8A20 calls to their call site
// (it has two in sub_1802EB620). Nothing else in this tree pulls this in.
#include <intrin.h>

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
  extern std::atomic<uint32_t> g_jobProbeLeafSkipLo;
  extern std::atomic<uint32_t> g_jobProbeLeafSkipHi;
  extern std::atomic<uint32_t> g_jobProbeSplitLo;
  extern std::atomic<uint32_t> g_jobProbeSplitHi;
  extern std::atomic<uint32_t> g_jobProbePlanesLo;
  extern std::atomic<uint32_t> g_jobProbePlanesHi;
  extern std::atomic<uint32_t> g_jobProbePoolLo;
  extern std::atomic<uint32_t> g_jobProbePoolHi;
  extern std::atomic<uint64_t> g_dispProbeA20Calls;
  extern std::atomic<uint64_t> g_dispProbeAllocCalls;
  extern std::atomic<uint64_t> g_dispProbeAllocFail;
  extern std::atomic<uint32_t> g_dispProbePendLo;
  extern std::atomic<uint32_t> g_dispProbePendHi;
  extern std::atomic<float> g_dispProbeFc000X;
  extern std::atomic<float> g_dispProbeFc000Y;
  extern std::atomic<float> g_dispProbeFc000Z;
  extern std::atomic<float> g_dispProbeFc000W;
  extern std::atomic<uint32_t> g_dispProbeSlotN;
  extern std::atomic<uint32_t> g_dispProbeSlotA1[32];
  extern std::atomic<uint32_t> g_dispProbeSlotA2[32];
  extern std::atomic<uint32_t> g_dispProbeSlotA3[32];
  extern std::atomic<uint32_t> g_dispProbeSlotRA[32];
  extern std::atomic<uint64_t> g_dispProbeEd480Calls;
  extern std::atomic<uint32_t> g_dispProbeEd480N;
  extern std::atomic<uint32_t> g_dispProbeEd480Area[8];

  // NV-DXVK [AreaSeed]: sub_1802EAD60's order list, read from EF090's wrapper.
  extern std::atomic<float>    g_areaSeedOrgX;
  extern std::atomic<float>    g_areaSeedOrgY;
  extern std::atomic<float>    g_areaSeedOrgZ;
  extern std::atomic<float>    g_areaSeedOrgW;
  extern std::atomic<uint32_t> g_areaSeedNAreas;
  extern std::atomic<uint32_t> g_areaSeedListLen;
  extern std::atomic<uint32_t> g_areaSeedN;
  extern std::atomic<uint32_t> g_areaSeedAreas[48];
  extern std::atomic<uint64_t> g_areaSeedCalls;
  extern std::atomic<uint32_t> g_areaSeedInstalled;
  extern std::atomic<uint32_t> g_areaSeedLive;
  extern std::atomic<uint32_t> g_areaSeedLiveN;
  extern std::atomic<uint32_t> g_areaSeedLiveAreas[48];
  extern std::atomic<uint32_t> g_areaSeedPending;
  extern std::atomic<uint32_t> g_clipDegenMaxBit;

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

    // The three traversal-governing globals — CULLING_BIBLE §13. Written by
    // sub_1802EF090 / sub_1802EB620 BEFORE the jobs are dispatched, so reading
    // them from inside a job reads the values actually in force for that
    // frame's traversal. See the block comment in rtx_camera_manager.cpp.
    constexpr std::uintptr_t kRvaPlaneCount     = 0x11FC0C0;  // 4, or 8 for the extra plane blocks
    constexpr std::uintptr_t kRvaLeafSkipThresh = 0x11FC110;  // leaf-skip threshold
    constexpr std::uintptr_t kRvaSplitThresh    = 0x11FC114;  // job-split subtree threshold

    // The record pool bump pointer (blocks used, cap 0xFFC = 4092) and EB620's
    // pending-area count. Both reset/seeded per EB620 call. See the block
    // comment in rtx_camera_manager.cpp — sub_1802E7C70 returns -1 once the
    // bump would exceed 0xFFC, and sub_1802E8A20 then does NOTHING AT ALL for
    // that area: no jobs, no mask bits, no log, no patchable reject.
    constexpr std::uintptr_t kRvaRecPoolUsed    = 0x11FC0DC;  // bump pointer, cap 0xFFC
    constexpr std::uintptr_t kRvaPendingAreas   = 0x11FC0D8;  // decremented per area consumed
    // xmmword_1811FC000 — xmm6 in sub_1802EAD60's BSP descent (0x2EAFA5) and
    // portal-crossing test (0x2EB0F1). Identity DISPUTED: read as the camera
    // origin, but sub_1802EF090 writes it from a2+0, and if a2 is a view
    // matrix that is row 0. Sampled raw to settle it.
    constexpr std::uintptr_t kRvaFrustum0       = 0x11FC000;

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

    // Set at install time once all three threshold globals have been proven
    // readable. NOT re-checked per job: they are fixed addresses in client.dll's
    // .data and cannot be unmapped while the module is loaded, and WvReadable
    // costs a VirtualQuery. That per-call VirtualQuery is exactly what got
    // dword_1813C0940 deleted from this probe (see below) — three of them on
    // ~180 job calls a frame would be 540 kernel transitions per frame to read
    // three dwords. Install-time validation plus the wrapper's SEH covers it.
    std::atomic<bool> s_wjGlobalsReadable{false};

    // Fold one dword of client.dll state into a per-frame [lo,hi] pair.
    inline void WjFoldGlobal(const std::uint8_t* client, std::uintptr_t rva,
                             std::atomic<std::uint32_t>& lo,
                             std::atomic<std::uint32_t>& hi) {
      const std::uint32_t v = *reinterpret_cast<const std::uint32_t*>(client + rva);
      WjAtomicMin(lo, v);
      WjAtomicMax(hi, v);
    }

    // Runs on the job threads. Must not log, allocate, or fault.
    void WorldJobSample(std::uint64_t jobIdx) {
      dxvk::tf2::g_jobProbeCalls.fetch_add(1, std::memory_order_relaxed);

      const std::uint8_t* client = s_wjClientBase.load(std::memory_order_acquire);
      if (client == nullptr) {
        dxvk::tf2::g_jobProbeBadReads.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      // Sampled FIRST, before anything that can early-out on the job index:
      // these three do not depend on jobIdx, and a run where every job had a
      // bad index would otherwise report no threshold data at all. All three
      // are folded unconditionally — no short-circuiting, or one unreadable
      // address would silently cost the other two their whole capture.
      if (s_wjGlobalsReadable.load(std::memory_order_relaxed)) {
        WjFoldGlobal(client, kRvaPlaneCount,
                     dxvk::tf2::g_jobProbePlanesLo, dxvk::tf2::g_jobProbePlanesHi);
        WjFoldGlobal(client, kRvaLeafSkipThresh,
                     dxvk::tf2::g_jobProbeLeafSkipLo, dxvk::tf2::g_jobProbeLeafSkipHi);
        WjFoldGlobal(client, kRvaSplitThresh,
                     dxvk::tf2::g_jobProbeSplitLo, dxvk::tf2::g_jobProbeSplitHi);
        WjFoldGlobal(client, kRvaRecPoolUsed,
                     dxvk::tf2::g_jobProbePoolLo, dxvk::tf2::g_jobProbePoolHi);
        // pend is NOT folded here any more — it counts down, so a max taken on
        // the job threads reports the maximum REMAINING, not the queue depth.
        // Moved to the [DispProbe] sub_1802E8A20 hook below, which samples it
        // at dispatch time. See rtx_camera_manager.cpp.
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

      // Validate the three threshold globals ONCE, here on the main thread, so
      // the job threads can read them with plain loads. Logged with their
      // install-time values because those are the pre-gameplay baseline: if a
      // capture never shows anything other than these, nothing is moving them.
      {
        auto* base = reinterpret_cast<std::uint8_t*>(client);
        const bool ok =
          WvReadable(base + kRvaPlaneCount,     sizeof(std::uint32_t)) &&
          WvReadable(base + kRvaLeafSkipThresh, sizeof(std::uint32_t)) &&
          WvReadable(base + kRvaSplitThresh,    sizeof(std::uint32_t)) &&
          WvReadable(base + kRvaRecPoolUsed,    sizeof(std::uint32_t)) &&
          WvReadable(base + kRvaPendingAreas,   sizeof(std::uint32_t));
        s_wjGlobalsReadable.store(ok, std::memory_order_relaxed);
        if (ok) {
          dxvk::Logger::warn(dxvk::str::format(
            "[JobProbe] globals readable — at install: planes=",
            *reinterpret_cast<const std::uint32_t*>(base + kRvaPlaneCount),
            " leafSkip=",
            *reinterpret_cast<const std::uint32_t*>(base + kRvaLeafSkipThresh),
            " split=",
            *reinterpret_cast<const std::uint32_t*>(base + kRvaSplitThresh),
            " pool=",
            *reinterpret_cast<const std::uint32_t*>(base + kRvaRecPoolUsed),
            " pend=",
            *reinterpret_cast<const std::uint32_t*>(base + kRvaPendingAreas),
            "  (pool cap is 4092 blocks)"));
        } else {
          // Not fatal: calls/recCntSum still measure. The other fields will read
          // [0,0] every line, which is a probe defect and must not be read as
          // "the value is zero" — CULLING_BIBLE §4a, the whole reason [OccProbe]
          // v1 had to be rewritten.
          dxvk::Logger::warn(
            "[JobProbe] client.dll globals NOT readable — planes/leafSkip/split/"
            "pool/pend will log [0,0]; treat those fields as ABSENT, not as zero");
        }
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
    // NV-DXVK [DispProbe] — the AREA DISPATCH path in sub_1802EB620.
    //
    // Three entry-point trampolines, all on the same gateway/trampoline scheme
    // as [JobProbe] above. See rtx_camera_manager.cpp for why these three and
    // for the deduction that bounds what they can prove.
    //
    //   sub_1802E8A20  dispatches one area's jobs. Counted, and dword_1811FC0D8
    //                  (pending areas) is folded HERE rather than on the job
    //                  threads, because it counts down.
    //   sub_1802ED900  builds the portal record. Returns -1 => EB620 drops the
    //                  area at 0x2EB8D0 with no dispatch.
    //   sub_1802E7C70  the record allocator. Returns -1 on pool exhaustion, and
    //                  sub_1802E8A20 then silently does nothing at all.
    //
    // The two -1 counters are the point. Both drops are invisible today: no
    // log, no reject branch, nothing any [CullOff] site can patch. allocFail is
    // expected to be 0 (the pool round measured 36/4092 peak) — it is counted
    // anyway so the refutation rests on the direct measurement rather than on
    // an inference from the bump pointer.
    // =======================================================================
    // ONLY FUNCTIONS THAT DECOMPILE ARE HOOKED HERE. A C wrapper commits to a
    // calling convention, and IDA's argument list for a function it could not
    // decompile is a guess, not a fact.
    //
    // sub_1802ED900 WAS hooked here on 2026-08-05 and FROZE THE GAME. It does
    // not decompile; its stack frame is nothing but _OWORD slots; it builds
    // portal planes. IDA reported one _DWORD arg and that was taken at face
    // value, so the wrapper was declared int __fastcall(unsigned int) — which
    // clobbers every volatile xmm register (and rdx/r8/r9) before calling
    // through. Do NOT re-add it as a C wrapper.
    // It is also UNNECESSARY: in EB620's loop an area dropped at 0x2EB8D0
    // (ED900 returned -1) never reaches sub_1802E8A20 at 0x2EB910, so
    //     areas dropped = queue depth - a20
    // which pend and a20 already give. If ED900 ever must be instrumented
    // directly, use a naked/asm thunk that preserves xmm0-5 and rdx/r8/r9, or a
    // mid-function detour at the CALL SITE (0x2EB8C5) where the convention is
    // visible, not an entry trampoline.
    using DispA20Fn   = std::int64_t(__fastcall*)(int, unsigned int, std::int64_t);
    using DispAllocFn = std::int64_t(__fastcall*)(int, int);
    using DispEd480Fn = std::int64_t(__fastcall*)(unsigned int);

    using DispEf090Fn = std::int64_t(__fastcall*)(unsigned int, std::int64_t);

    DispA20Fn   s_origDispA20   = nullptr;
    DispAllocFn s_origDispAlloc = nullptr;
    DispEd480Fn s_origDispEd480 = nullptr;
    DispEf090Fn s_origDispEf090 = nullptr;

    constexpr std::uintptr_t kRvaDispA20   = 0x2E8A20;  // sub_1802E8A20 (decompiles, 3 args)
    constexpr std::uintptr_t kRvaDispAlloc = 0x2E7C70;  // sub_1802E7C70 (decompiles, 2 args)
    constexpr std::uintptr_t kRvaDispEd480 = 0x2ED480;  // sub_1802ED480 (decompiles, 1 arg)
    // [AreaSeed]. sub_1802EF090 is EB620's DIRECT CALLER and it DECOMPILES —
    // `__int64 __fastcall(unsigned int a1, __int64 a2)`, read from the body, not
    // guessed. That is what makes this hook legal where one on sub_1802EAD60 or
    // sub_1802EB620 is not: neither decompiles, and the last wrapper committed to
    // a guessed convention froze the game.
    constexpr std::uintptr_t kRvaDispEf090   = 0x2EF090;   // sub_1802EF090
    constexpr std::uintptr_t kRvaAreaOrder   = 0x11FE920;  // word_1811FE920, the order list
    constexpr std::uintptr_t kRvaAreaCount   = 0x1748D8C;  // dword_181748D8C, its bound
    // dword_1811FF91C, the per-area record selector. memset to -1 at 0x2EB765
    // on entry, and each area's entry is consumed back to -1 at 0x2EB88A when
    // the queue loop VISITS it. So an entry still holding a non -1 value when
    // EB620 returns is an area that was ENQUEUED AND NEVER VISITED — which is
    // exactly the distinction the pitch/yaw well needs and it costs no new hook.
    constexpr std::uintptr_t kRvaAreaSelector = 0x11FF91C;
    // dword_1811FC0D8, the pending count. Should be 0 at exit; anything else
    // means the loop left work on the table.
    constexpr std::uintptr_t kRvaAreaPending  = 0x11FC0D8;
    // Sanity bound on dword_181748D8C before it is used as a loop count. The map
    // this is measured on reports ~180 areas; anything past 8192 means the read
    // landed on garbage (module not fully loaded, or the wrong build) and the
    // sentinel fill is skipped rather than scribbling over unrelated memory.
    constexpr std::uint32_t  kAreaCountMax   = 8192u;

    // Prologue sizes differ per target and are NOT interchangeable — each is
    // the smallest whole-instruction run of at least 5 bytes, verified against
    // the expected bytes at install so a different game build is skipped rather
    // than corrupted:
    //   E8A20  48 89 5C 24 08              mov [rsp+8], rbx              = 5
    //   E7C70  8D 04 91 / 44 8B D2         lea eax,[rcx+rdx*4]; mov r10d,edx = 6
    // Both are position independent, so the stolen bytes need no relocation.
    constexpr std::size_t kDispA20PrologueSize   = 5;
    constexpr std::size_t kDispAllocPrologueSize = 6;
    constexpr std::size_t kDispMaxPrologueSize   = 8;

    std::atomic<std::uint8_t*> s_dispClientBase{nullptr};

    // Generic installer for the three: same two-hop gateway as [JobProbe] (the
    // outbound jmp cannot reach d3d11.dll directly, hence the near page).
    bool DispInstallOne(const char* tag, std::uintptr_t rva, std::size_t prologueSize,
                        const std::uint8_t* expect, void* wrapper, void** origOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* target = reinterpret_cast<std::uint8_t*>(client) + rva;

      if (std::memcmp(target, expect, prologueSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[DispProbe] ", tag, " prologue mismatch at client.dll+0x", std::hex, rva,
          std::dec, " — not this build, hook skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(reinterpret_cast<std::uint8_t*>(client));
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[DispProbe] ", tag, ": no page within rel32 range of client.dll"));
        return false;
      }

      std::uint8_t* gate = page;
      gate[0] = 0xFF; gate[1] = 0x25;
      gate[2] = 0x00; gate[3] = 0x00; gate[4] = 0x00; gate[5] = 0x00;
      const auto absWrapper = reinterpret_cast<std::uint64_t>(wrapper);
      std::memcpy(gate + 6, &absWrapper, sizeof(absWrapper));

      std::uint8_t* tramp = page + 16;
      std::memcpy(tramp, target, prologueSize);
      std::uint8_t* jb = tramp + prologueSize;
      jb[0] = 0xFF; jb[1] = 0x25; jb[2] = 0x00; jb[3] = 0x00; jb[4] = 0x00; jb[5] = 0x00;
      const auto retAddr = reinterpret_cast<std::uint64_t>(target + prologueSize);
      std::memcpy(jb + 6, &retAddr, sizeof(retAddr));
      *origOut = tramp;

      s_dispClientBase.store(reinterpret_cast<std::uint8_t*>(client), std::memory_order_release);

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(gate)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[DispProbe] ", tag, ": gateway out of rel32 range, hook skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, prologueSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[DispProbe] ", tag, ": VirtualProtect failed"));
        return false;
      }
      // 5-byte jmp, then NOP out any remaining stolen bytes so the patched
      // region stays decodable if anything ever disassembles it.
      std::uint8_t stub[kDispMaxPrologueSize];
      std::memset(stub, 0x90, sizeof(stub));
      stub[0] = 0xE9;
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, prologueSize);
      DWORD tmp = 0;
      VirtualProtect(target, prologueSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, prologueSize);

      dxvk::Logger::warn(dxvk::str::format(
        "[DispProbe] ", tag, " INSTALLED at client.dll+0x", std::hex, rva, std::dec));
      return true;
    }

    std::int64_t __fastcall DispA20Wrapper(int a1, unsigned int a2, std::int64_t a3) {
      dxvk::tf2::g_dispProbeA20Calls.fetch_add(1, std::memory_order_relaxed);

      // [AreaDump]: record WHICH area, not just how many. a3 is the seed BSP
      // node (qword_181748D58 + 32*a3) and is the area's identity. Slot count
      // is bounded at 8; slotN keeps counting past that so an overflow is
      // visible rather than silent. Plain stores — this is a diagnostic
      // snapshot, not a queue, and a torn read costs one frame of one line.
      {
        const std::uint32_t s =
          dxvk::tf2::g_dispProbeSlotN.fetch_add(1, std::memory_order_relaxed);
        if (s < 32u) {
          dxvk::tf2::g_dispProbeSlotA1[s].store(static_cast<std::uint32_t>(a1),
                                                std::memory_order_relaxed);
          dxvk::tf2::g_dispProbeSlotA2[s].store(static_cast<std::uint32_t>(a2),
                                                std::memory_order_relaxed);
          dxvk::tf2::g_dispProbeSlotA3[s].store(static_cast<std::uint32_t>(a3),
                                                std::memory_order_relaxed);
          // WHICH CALL SITE. sub_1802E8A20 has two callers in EB620 —
          // 0x2EB910 (the queue loop, fc000-gated, position-only) and
          // 0x2EC937 (never read, and its region uses the frustum SIDE
          // planes). a20 has been merging both all along, which is why the
          // path that measured constant and the output that fell could both
          // be true. This is an entry trampoline, so the return address on
          // the stack is the caller's — expect 0x2eb915 / 0x2ec93c.
          const std::uint8_t* cb =
            s_dispClientBase.load(std::memory_order_acquire);
          const auto ra = reinterpret_cast<std::uintptr_t>(_ReturnAddress());
          const std::uint32_t raRva =
            (cb != nullptr && ra > reinterpret_cast<std::uintptr_t>(cb))
              ? static_cast<std::uint32_t>(ra - reinterpret_cast<std::uintptr_t>(cb))
              : 0u;
          dxvk::tf2::g_dispProbeSlotRA[s].store(raRva, std::memory_order_relaxed);
        }
      }
      // Sampled at entry: EB620 has already decremented the pending count for
      // this area at 0x2EB8AF, so this reads (queue depth - 1) on the frame's
      // first dispatch. Max over the frame is therefore the best available
      // estimate of how many areas were queued.
      const std::uint8_t* client = s_dispClientBase.load(std::memory_order_acquire);
      if (client != nullptr && s_wjGlobalsReadable.load(std::memory_order_relaxed)) {
        __try {
          WjFoldGlobal(client, kRvaPendingAreas,
                       dxvk::tf2::g_dispProbePendLo, dxvk::tf2::g_dispProbePendHi);
          // The four raw floats at 0x11FC000 — xmm6 in sub_1802EAD60's two
          // gates. Constant under yaw => camera origin; changing => the
          // identification was wrong and EAD60 is view-dependent after all.
          // See rtx_camera_manager.cpp for why this decides the argument.
          // Sampled at dispatch time, which is after sub_1802EF090 has written
          // it and inside the EB620 call that consumes it.
          const auto* fc = reinterpret_cast<const float*>(client + kRvaFrustum0);
          dxvk::tf2::g_dispProbeFc000X.store(fc[0], std::memory_order_relaxed);
          dxvk::tf2::g_dispProbeFc000Y.store(fc[1], std::memory_order_relaxed);
          dxvk::tf2::g_dispProbeFc000Z.store(fc[2], std::memory_order_relaxed);
          dxvk::tf2::g_dispProbeFc000W.store(fc[3], std::memory_order_relaxed);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
        }
      }
      return s_origDispA20(a1, a2, a3);
    }

    // [Ed480Probe] — sub_1802ED480 is THE DYNAMIC AREA ENQUEUE.
    //
    // WHY IT MATTERS. [AreaDump] showed exactly five areas, of which two (BSP
    // nodes 124, 178) are present at every yaw and three (127, then 125/149)
    // drop out progressively past ~140deg. sub_1802EAD60 produces the
    // unconditional pair and is position-only (fc000 measured constant on 671
    // frames). The only other enqueue in the whole path is this function —
    // dword_1811FF91C[a1] = rec; ++dword_1811FC0D8 — and it has NO code xrefs:
    // it is reached solely through fn-ptr table slots 0x183C54E44/0x183C54E50,
    // i.e. it runs as a job. So the three view-dependent areas come from here.
    //
    // SAFE TO WRAP, unlike sub_1802ED900: this one DECOMPILES, the signature
    // is read from the body (one unsigned int, no SIMD args), and the whole
    // function is 0x120 bytes of plain integer work. That is the standing rule
    // — only wrap what decompiles.
    //
    // THE CALLER IS THE POINT. a1 tells us WHICH areas get enqueued; the
    // BACKTRACE names whichever function decided a portal was visible, and
    // that decision is the actual fix site. Captured once (one-shot) because
    // it is the identity that is wanted, not a per-call histogram, and because
    // this runs on job threads where formatting a string every call would
    // perturb what is being measured.
    std::atomic<bool> s_ed480StackCaptured{false};

    std::int64_t __fastcall DispEd480Wrapper(unsigned int a1) {
      dxvk::tf2::g_dispProbeEd480Calls.fetch_add(1, std::memory_order_relaxed);

      // Which area. Same bounded-slot scheme as [AreaDump].
      const std::uint32_t s =
        dxvk::tf2::g_dispProbeEd480N.fetch_add(1, std::memory_order_relaxed);
      if (s < 8u)
        dxvk::tf2::g_dispProbeEd480Area[s].store(a1, std::memory_order_relaxed);

      // One-shot backtrace: name the caller that decided this area is visible.
      bool expected = false;
      if (s_ed480StackCaptured.compare_exchange_strong(expected, true,
                                                       std::memory_order_relaxed)) {
        void* frames[16];
        const USHORT n = RtlCaptureStackBackTrace(0, 16, frames, nullptr);
        struct Mod { const char* name; std::uint64_t base; };
        Mod mods[4] = {
          { "cli",   reinterpret_cast<std::uint64_t>(GetModuleHandleA("client.dll")) },
          { "eng",   reinterpret_cast<std::uint64_t>(GetModuleHandleA("engine.dll")) },
          { "d3d11", reinterpret_cast<std::uint64_t>(GetModuleHandleA("d3d11.dll")) },
          { "mat",   reinterpret_cast<std::uint64_t>(GetModuleHandleA("materialsystem_dx11.dll")) },
        };
        constexpr std::uint64_t kWindow = 0x40000000ull;
        std::string st;
        st.reserve(512);
        for (USHORT k = 0; k < n && k < 16; ++k) {
          const std::uint64_t addr = reinterpret_cast<std::uint64_t>(frames[k]);
          const char* mod = "?";
          std::uint64_t rva = addr, bestBase = 0;
          for (int m = 0; m < 4; ++m) {
            if (mods[m].base && addr >= mods[m].base && mods[m].base > bestBase
                && addr < mods[m].base + kWindow) {
              bestBase = mods[m].base; mod = mods[m].name; rva = addr - mods[m].base;
            }
          }
          if (!st.empty()) st += " | ";
          st += dxvk::str::format(mod, "+0x", std::hex, rva, std::dec);
        }
        dxvk::Logger::warn(dxvk::str::format(
          "[Ed480Probe] FIRST CALL area=", a1, " stack: ", st));
      }

      return s_origDispEd480(a1);
    }

    // [AreaSeed] — what sub_1802EAD60 actually seeded, without hooking it.
    //
    // EF090's body (decompiled) is, in order: read a2+0/16/32/48, build the four
    // frustum side planes, `xmmword_1811FC000 = *(_OWORD*)a2`, write fc010-fc070,
    // `dword_1811FC0C0 = 4`, then `sub_1802EB620(a1, a2, 0)`. So at ENTRY, a2+0
    // is exactly the value EAD60 will consume, and it is the only place that
    // value can be observed: EB620's tail overwrites the global at 0x2ECB2F
    // before returning.
    //
    // The sentinel fill recovers the order list's extent without knowing EAD60's
    // return value. It writes only word_1811FE920[0, nAreas), which EAD60 then
    // overwrites from the top down; the queue loop reads nothing below its
    // cursor, so untouched sentinels are never observed by the engine.
    std::int64_t __fastcall DispEf090Wrapper(unsigned int a1, std::int64_t a2) {
      dxvk::tf2::g_areaSeedCalls.fetch_add(1, std::memory_order_relaxed);

      const std::uint8_t* client = s_dispClientBase.load(std::memory_order_acquire);
      std::uint32_t nAreas = 0;

      __try {
        if (a2 != 0) {
          const auto* org = reinterpret_cast<const float*>(a2);
          dxvk::tf2::g_areaSeedOrgX.store(org[0], std::memory_order_relaxed);
          dxvk::tf2::g_areaSeedOrgY.store(org[1], std::memory_order_relaxed);
          dxvk::tf2::g_areaSeedOrgZ.store(org[2], std::memory_order_relaxed);
          dxvk::tf2::g_areaSeedOrgW.store(org[3], std::memory_order_relaxed);
        }
        if (client != nullptr) {
          nAreas = *reinterpret_cast<const std::uint32_t*>(client + kRvaAreaCount);
          if (nAreas != 0u && nAreas <= kAreaCountMax) {
            auto* list = reinterpret_cast<std::uint16_t*>(
              const_cast<std::uint8_t*>(client) + kRvaAreaOrder);
            for (std::uint32_t i = 0; i < nAreas; ++i)
              list[i] = 0xFFFFu;
          } else {
            nAreas = 0;
          }
          dxvk::tf2::g_areaSeedNAreas.store(
            *reinterpret_cast<const std::uint32_t*>(client + kRvaAreaCount),
            std::memory_order_relaxed);
        }
      } __except (EXCEPTION_EXECUTE_HANDLER) {
        nAreas = 0;
      }

      const std::int64_t ret = s_origDispEf090(a1, a2);

      if (nAreas != 0u) {
        __try {
          const auto* list = reinterpret_cast<const std::uint16_t*>(client + kRvaAreaOrder);
          std::uint32_t len = 0, kept = 0;
          for (std::uint32_t i = 0; i < nAreas; ++i) {
            if (list[i] == 0xFFFFu)
              continue;
            ++len;
            // 48 to cover listLen (30 on this map) with headroom. At 16 this
            // recorded barely half the seed list while printing listLen=30
            // beside it, which is how the seed set was read as "identical in
            // both states" — the differing entries were in the 14 never stored.
            if (kept < 48u)
              dxvk::tf2::g_areaSeedAreas[kept++].store(list[i], std::memory_order_relaxed);
          }
          dxvk::tf2::g_areaSeedListLen.store(len, std::memory_order_relaxed);
          dxvk::tf2::g_areaSeedN.store(kept, std::memory_order_relaxed);

          // Areas left holding a selector: enqueued, never visited. See
          // kRvaAreaSelector — this separates "the crossing into it never
          // happened" from "it was queued and the loop lost it", which is the
          // one thing the 2D well has not distinguished.
          const auto* sel = reinterpret_cast<const std::int32_t*>(client + kRvaAreaSelector);
          std::uint32_t live = 0, liveKept = 0;
          for (std::uint32_t i = 0; i < nAreas; ++i) {
            if (sel[i] == -1)
              continue;
            ++live;
            if (liveKept < 48u)
              dxvk::tf2::g_areaSeedLiveAreas[liveKept++].store(i, std::memory_order_relaxed);
          }
          dxvk::tf2::g_areaSeedLive.store(live, std::memory_order_relaxed);
          dxvk::tf2::g_areaSeedLiveN.store(liveKept, std::memory_order_relaxed);
          dxvk::tf2::g_areaSeedPending.store(
            *reinterpret_cast<const std::uint32_t*>(client + kRvaAreaPending),
            std::memory_order_relaxed);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
        }
      }
      return ret;
    }

    std::int64_t __fastcall DispAllocWrapper(int a1, int a2) {
      dxvk::tf2::g_dispProbeAllocCalls.fetch_add(1, std::memory_order_relaxed);
      const std::int64_t ret = s_origDispAlloc(a1, a2);
      // Compared as a DWORD because the callers test (_DWORD)result != -1 and
      // sub_1802E7C70 returns 0xFFFFFFFF zero-extended, not a sign-extended -1.
      if (static_cast<std::uint32_t>(ret) == 0xFFFFFFFFu)
        dxvk::tf2::g_dispProbeAllocFail.fetch_add(1, std::memory_order_relaxed);
      return ret;
    }

    bool DoInstallDispProbe() {
      static constexpr std::uint8_t kA20[kDispA20PrologueSize] = {
        0x48, 0x89, 0x5C, 0x24, 0x08,                     // mov [rsp+arg_0], rbx
      };
      static constexpr std::uint8_t kAlloc[kDispAllocPrologueSize] = {
        0x8D, 0x04, 0x91,                                 // lea  eax, [rcx+rdx*4]
        0x44, 0x8B, 0xD2,                                 // mov  r10d, edx
      };
      static_assert(kDispA20PrologueSize   <= kDispMaxPrologueSize, "stub buffer");
      static_assert(kDispAllocPrologueSize <= kDispMaxPrologueSize, "stub buffer");

      bool any = false;
      any |= DispInstallOne("E8A20", kRvaDispA20, kDispA20PrologueSize, kA20,
                            reinterpret_cast<void*>(&DispA20Wrapper),
                            reinterpret_cast<void**>(&s_origDispA20));
      // sub_1802ED480: mov [rsp+8], rbx  = 48 89 5C 24 08, exactly 5.
      static constexpr std::uint8_t kEd480[5] = {
        0x48, 0x89, 0x5C, 0x24, 0x08,
      };
      static_assert(sizeof(kEd480) <= kDispMaxPrologueSize, "stub buffer");
      any |= DispInstallOne("ED480", kRvaDispEd480, sizeof(kEd480), kEd480,
                            reinterpret_cast<void*>(&DispEd480Wrapper),
                            reinterpret_cast<void**>(&s_origDispEd480));
      any |= DispInstallOne("E7C70", kRvaDispAlloc, kDispAllocPrologueSize, kAlloc,
                            reinterpret_cast<void*>(&DispAllocWrapper),
                            reinterpret_cast<void**>(&s_origDispAlloc));
      // sub_1802EF090: push rbx (2) + sub rsp,70h (4) = 6, the smallest whole
      // instruction run of at least 5. Both position independent, so the stolen
      // bytes need no relocation.
      static constexpr std::uint8_t kEf090[6] = {
        0x40, 0x53,                                       // push rbx
        0x48, 0x83, 0xEC, 0x70,                           // sub  rsp, 70h
      };
      static_assert(sizeof(kEf090) <= kDispMaxPrologueSize, "stub buffer");
      const bool ef090 =
        DispInstallOne("EF090", kRvaDispEf090, sizeof(kEf090), kEf090,
                       reinterpret_cast<void*>(&DispEf090Wrapper),
                       reinterpret_cast<void**>(&s_origDispEf090));
      // Recorded because an uninstalled hook and a never-called function both
      // report zero — the [OccProbe] v1 / ed900Inst lesson.
      dxvk::tf2::g_areaSeedInstalled.store(ef090 ? 1u : 0u, std::memory_order_relaxed);
      any |= ef090;
      return any;
    }

    std::once_flag s_onceDispProbe;
    std::atomic<bool> s_dispProbeInstalled{false};

    // =======================================================================
    // NV-DXVK [Ed900Probe] — "is sub_1802ED900 called at all?"
    //
    // THE ONE FORK LEFT. sub_1802ED900 reads xmmword_1811FC030 (the camera
    // FORWARD, from which EF090 derives all four frustum side planes) at four
    // sites, so it IS view-direction dependent, and a -1 from it drops a whole
    // area at 0x2EB8D0 — no dispatch, no jobs, no mask bits, no TLAS geometry.
    // That fits every measurement. BUT it may never run: EB620 tests
    //     0x2EB8B8  cmp dword ptr [rdi+4], -1
    //     0x2EB8C0  jz  loc_1802EB8EB          ; skip ED900, use record as-is
    // and sub_1802E7C70 sets rec[+4] = -1 on every fresh allocation. If live
    // records keep that value, the branch is ALWAYS taken and ED900 is dead
    // code on this path. Supporting worry from the last capture: alloc/a20
    // stays ~3 across yaw, which looks more like fewer areas ENQUEUED than
    // areas enqueued-then-dropped.
    //   count > 0, tracking the yaw collapse => ED900 is the mechanism, and
    //     the fix is 0x2EB8C0: 74 29 -> EB 29 (force the engine's own skip
    //     path; rdi is already a valid record there, so no wild selector).
    //   count ~ 0 => ED900 is eliminated and the loss is on the ENQUEUE side.
    //     Go back to sub_1802EAD60's second phase (0x2EB00C onward), which is
    //     only half read.
    //
    // WHY THIS IS NOT A C WRAPPER. Wrapping ED900 as int __fastcall(unsigned)
    // is what FROZE THE GAME on 2026-08-05: it does not decompile, IDA's
    // argument list was a guess, and a C wrapper clobbers the volatile xmm
    // registers a SIMD plane-builder needs. This island executes ONE
    // instruction of its own — `lock inc` — which writes only FLAGS, and flags
    // are dead at a function entry boundary. It then runs the stolen prologue
    // and jumps back. No call, no shadow space, no register assumption, no
    // convention committed to. It cannot perturb ED900 even if every guess
    // about its signature is wrong.
    //
    // ISLAND LAYOUT (built by hand; offsets are load-bearing):
    //   +0x00  F0 48 FF 05 28 00 00 00   lock inc qword ptr [rip+0x28] -> +0x30
    //   +0x08  48 8B C4                  mov  rax, rsp          (stolen)
    //   +0x0B  48 89 58 10               mov  [rax+10h], rbx    (stolen)
    //   +0x0F  FF 25 00 00 00 00         jmp  qword ptr [rip+0]
    //   +0x15  <qword>                   = client + 0x2ED907 (past the steal)
    //   +0x30  <qword>                   the counter itself
    // rip after the lock inc is +0x08, target +0x30, so disp = 0x28.
    // =======================================================================
    // Generic counter-island installer. `stealSize` MUST be a whole number of
    // instructions and >= 5; the caller supplies the exact expected bytes, so a
    // different game build fails the memcmp and is skipped rather than
    // corrupted. Every stolen prologue used here is position independent, so
    // nothing needs relocating.
    //
    // Island layout (offsets load-bearing; max steal 8 keeps +0x15 clear of the
    // return-address qword and the counter at +0x30):
    //   +0x00              F0 48 FF 05 <disp32>   lock inc qword [rip+disp]
    //   +0x08              <stolen prologue>
    //   +0x08+steal        FF 25 00 00 00 00      jmp qword ptr [rip+0]
    //   +0x0E+steal        <qword> return address
    //   +0x30              <qword> counter
    // rip after the lock inc is +0x08, so disp = 0x30 - 0x08 = 0x28.
    constexpr std::size_t kIslandMaxSteal = 8;

    bool InstallCounterIsland(const char* tag, std::uintptr_t rva,
                              std::size_t stealSize, const std::uint8_t* expect,
                              volatile std::uint64_t** counterOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* target = reinterpret_cast<std::uint8_t*>(client) + rva;

      if (std::memcmp(target, expect, stealSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] prologue mismatch at client.dll+0x", std::hex, rva,
          std::dec, " — not this build, hook skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(reinterpret_cast<std::uint8_t*>(client));
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, 0x40);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      const std::int32_t lockDisp = 0x28;
      std::memcpy(page + o, &lockDisp, 4); o += 4;              // o == 0x08
      std::memcpy(page + o, expect, stealSize); o += stealSize;
      page[o++] = 0xFF; page[o++] = 0x25;
      const std::int32_t zero = 0;
      std::memcpy(page + o, &zero, 4); o += 4;
      const auto retAddr = reinterpret_cast<std::uint64_t>(target + stealSize);
      std::memcpy(page + o, &retAddr, sizeof(retAddr));
      const std::uint64_t zero64 = 0;
      std::memcpy(page + 0x30, &zero64, sizeof(zero64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] island out of rel32 range, hook skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, stealSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      std::uint8_t stub[kIslandMaxSteal];
      std::memset(stub, 0x90, sizeof(stub));
      stub[0] = 0xE9;
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, stealSize);
      DWORD tmp = 0;
      VirtualProtect(target, stealSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, stealSize);

      // Published only after the patch lands, so a reader can never see a
      // counter address for a hook that failed halfway.
      *counterOut = reinterpret_cast<volatile std::uint64_t*>(page + 0x30);

      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] INSTALLED at client.dll+0x", std::hex, rva,
        " island=0x", reinterpret_cast<std::uintptr_t>(page), std::dec,
        " (counter-only: lock inc + stolen prologue + jmp back)"));
      return true;
    }

    // =======================================================================
    // Counter island, variant for a block that ENDS IN AN UNCONDITIONAL
    // `jmp rel32`.
    //
    // WHY A VARIANT IS NEEDED. InstallCounterIsland above copies the stolen
    // bytes verbatim and jumps back to target+stealSize. That is only correct
    // for position-independent bytes, which every prologue it is used on is. A
    // `jmp rel32` is NOT position independent: replayed from the island its
    // displacement resolves against the island's rip and lands in the middle of
    // nowhere. Copying it verbatim would not fault at install — it would fault
    // the first time the branch was taken, which is the worst possible failure
    // mode for a diagnostic.
    //
    // The fix is not to relocate it. The tail jmp is unconditional, so its
    // destination is a constant: replay only the leading `replaySize` bytes and
    // finish with the same indirect jmp the base installer already uses, loaded
    // with the ABSOLUTE continuation address. Nothing needs a displacement.
    //
    //   +0x00              F0 48 FF 05 28 00 00 00   lock inc qword [rip+0x28]
    //   +0x08              <replaySize bytes>        leading, position-independent
    //   +0x08+replay       FF 25 00 00 00 00         jmp qword ptr [rip+0]
    //   +0x0E+replay       <qword> continuation      = client + continueRva
    //   +0x30              <qword> counter
    //
    // The caller still supplies the FULL expected bytes for all stealSize bytes,
    // so the rel32 itself is verified — a different build whose branch target
    // moved fails the memcmp and is skipped rather than silently miscounted.
    bool InstallCounterIslandTailJmp(const char* tag, std::uintptr_t rva,
                                     std::size_t stealSize, std::size_t replaySize,
                                     const std::uint8_t* expect,
                                     std::uintptr_t continueRva,
                                     volatile std::uint64_t** counterOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* target = reinterpret_cast<std::uint8_t*>(client) + rva;

      if (std::memcmp(target, expect, stealSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] byte mismatch at client.dll+0x", std::hex, rva,
          std::dec, " — not this build, hook skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(reinterpret_cast<std::uint8_t*>(client));
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, 0x40);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      const std::int32_t lockDisp = 0x28;
      std::memcpy(page + o, &lockDisp, 4); o += 4;              // o == 0x08
      std::memcpy(page + o, expect, replaySize); o += replaySize;
      page[o++] = 0xFF; page[o++] = 0x25;
      const std::int32_t zero = 0;
      std::memcpy(page + o, &zero, 4); o += 4;
      const auto contAddr =
        reinterpret_cast<std::uint64_t>(reinterpret_cast<std::uint8_t*>(client) + continueRva);
      std::memcpy(page + o, &contAddr, sizeof(contAddr));
      const std::uint64_t zero64 = 0;
      std::memcpy(page + 0x30, &zero64, sizeof(zero64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] island out of rel32 range, hook skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, stealSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      std::uint8_t stub[kIslandMaxSteal];
      std::memset(stub, 0x90, sizeof(stub));
      stub[0] = 0xE9;
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, stealSize);
      DWORD tmp = 0;
      VirtualProtect(target, stealSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, stealSize);

      *counterOut = reinterpret_cast<volatile std::uint64_t*>(page + 0x30);

      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] INSTALLED at client.dll+0x", std::hex, rva,
        " island=0x", reinterpret_cast<std::uintptr_t>(page),
        " cont=client.dll+0x", continueRva, std::dec,
        " (counter-only: lock inc + ", replaySize, " replayed bytes + abs jmp)"));
      return true;
    }

    // =======================================================================
    // Counting trampoline for a CONDITIONAL branch.
    //
    // WHY NOT AN ISLAND AT THE TARGET. Both branches this is used on jump to
    // loc_1802EC8ED, which has several other predecessors (0x2EBCDC before it
    // was retargeted, 0x2EC702, 0x2EC8A6, 0x2EC8AF, and the fallthrough at
    // 0x2EC8E5). A counter placed at the target would merge all of them and
    // report a number that means nothing — the [OccProbe] v1 failure shape.
    //
    // WHY NOT THE UNCONDITIONAL INSTALLERS ABOVE. Both of those overwrite the
    // site with `E9 rel32`. Doing that to a conditional branch would make it
    // unconditional, i.e. it would CHANGE BEHAVIOUR while pretending to measure
    // it. That is the one thing a diagnostic must never do.
    //
    // WHAT THIS DOES INSTEAD. The branch keeps its opcode, its length and its
    // condition; only the 4-byte displacement is rewritten to point at a stub
    // that increments a counter and jumps on to where the branch always went.
    // Semantically a no-op — the same instruction, taken under exactly the same
    // flags, arriving at exactly the same place. Same edit shape as the site-13
    // CullOff patch, which is a retarget for the same reason.
    //
    //   +0x00  F0 48 FF 05 28 00 00 00   lock inc qword [rip+0x28] -> +0x30
    //   +0x08  FF 25 00 00 00 00         jmp qword ptr [rip+0]
    //   +0x0E  <qword> original target   = client + origTargetRva
    //   +0x30  <qword> counter
    //
    // Flags: `lock inc` writes them, but the branch's own condition was already
    // consumed by the branch itself, and loc_1802EC8ED opens with
    // `mov rdx, cs:qword_181748CF8`, which does not read flags.
    //
    // branchSize must be the whole instruction and dispOffset where its rel32
    // begins (2 for the 0F 8x form, 1 for the EB/7x short forms, which this does
    // not support — a short branch cannot reach an allocated page).
    bool InstallBranchCounter(const char* tag, std::uintptr_t rva,
                              std::size_t branchSize, std::size_t dispOffset,
                              const std::uint8_t* expect,
                              std::uintptr_t origTargetRva,
                              volatile std::uint64_t** counterOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* base   = reinterpret_cast<std::uint8_t*>(client);
      auto* target = base + rva;

      if (std::memcmp(target, expect, branchSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] byte mismatch at client.dll+0x", std::hex, rva,
          std::dec, " — not this build, counter skipped"));
        return false;
      }

      // Cross-check the encoded destination against what the caller claims, so a
      // build whose branch survived the memcmp but points somewhere else cannot
      // be silently miscounted.
      std::int32_t curDisp = 0;
      std::memcpy(&curDisp, expect + dispOffset, sizeof(curDisp));
      const std::uintptr_t encodedTarget = rva + branchSize + curDisp;
      if (encodedTarget != origTargetRva) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] branch at client.dll+0x", std::hex, rva, " targets 0x",
          encodedTarget, ", expected 0x", origTargetRva, std::dec,
          " — counter skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(base);
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, 0x40);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      const std::int32_t lockDisp = 0x28;
      std::memcpy(page + o, &lockDisp, 4); o += 4;              // o == 0x08
      page[o++] = 0xFF; page[o++] = 0x25;
      const std::int32_t zero = 0;
      std::memcpy(page + o, &zero, 4); o += 4;
      const auto contAddr = reinterpret_cast<std::uint64_t>(base + origTargetRva);
      std::memcpy(page + o, &contAddr, sizeof(contAddr));
      const std::uint64_t zero64 = 0;
      std::memcpy(page + 0x30, &zero64, sizeof(zero64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) +
         static_cast<std::int64_t>(branchSize));
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub out of rel32 range, counter skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target + dispOffset, 4, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(target + dispOffset, &rel32, sizeof(rel32));
      DWORD tmp = 0;
      VirtualProtect(target + dispOffset, 4, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, branchSize);

      *counterOut = reinterpret_cast<volatile std::uint64_t*>(page + 0x30);

      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] INSTALLED at client.dll+0x", std::hex, rva,
        " stub=0x", reinterpret_cast<std::uintptr_t>(page),
        " -> client.dll+0x", origTargetRva, std::dec,
        " (branch retarget: condition and length unchanged)"));
      return true;
    }

    // =======================================================================
    // [ClipDegen] PORTAL RECORDER — identity, not volume.
    //
    // WHY THE COUNTER WAS NOT ENOUGH, and this is a correction of an earlier
    // reading in this file: clipDegen was dismissed as "anti-correlated with
    // the collapse, therefore not the gate". That does not follow. It is a
    // per-frame total over every portal of every area, while the mechanism is
    // ONE crossing failing and cascading — areas 113/141/148/153 appear and
    // vanish as a group. One portal dying moves a total of 17-40 by one, which
    // is invisible, and the total falls in the well simply because there are
    // fewer areas left to iterate. Same statistic-vs-mechanism mistake that had
    // ed900's flat call count reading as an acquittal.
    //
    // So record WHICH portal is abandoned. At 0x2EC675 the neighbour area has
    // not been read yet (that is 0x2EC716), but the portal index is live in
    // var_7B8, which IDA renders as [rbp+8B0h+var_7B8] = [rbp+0xF8]. It holds
    // portalIdx*3 (0x2EB93F stores rcx = eax*3), and the consumer at 0x2EC708
    // does [portalTable + rcx*4 + 6], so neighbourArea =
    // *(uint16*)(qword_181748D00 + bit*4 + 6) for a recorded bit.
    //
    // A BIT-SET, NOT A RING. Setting bit[portalIdx*3] needs no index variable
    // and therefore no second scratch register: `bts [rip+disp], rax` takes the
    // register as a bit-string offset and addresses the right qword itself. rax
    // is masked to 0xFFF first so the write can never leave the 512-byte table
    // even if the slot is garbage.
    //
    // COST: two scratch registers, saved and restored, and two memory writes.
    // Flags (from `and`, `bts` and the two `cmp`s) are dead at loc_1802EC8ED,
    // which opens with `mov rdx, cs:qword_181748CF8`. This is the first probe
    // here that reads through rbp — legitimate, since sub_1802EB620 keeps a
    // real frame pointer and var_7B8 is live across the whole portal iteration
    // — but it is also the first thing to switch off if the game misbehaves.
    //
    // Windows x64 has no red zone, so the two pushes are simply 16 bytes of
    // stack that the pops give straight back; nothing is called in between and
    // nothing can fault, so there is no unwind to get wrong.
    //
    //   0x00  F0 48 FF 05 58 00 00 00   lock inc qword [rip+0x58] -> +0x60
    //   0x08  50                        push rax
    //   0x09  51                        push rcx
    //   0x0A  48 8B 85 F8 00 00 00      mov  rax, [rbp+0xF8]      (var_7B8)
    //   0x11  48 25 FF 3F 00 00         and  rax, 3FFFh
    //   0x17  48 0F AB 05 E1 00 00 00   bts  [rip+0xE1], rax  -> +0x100
    //   0x1F  4C 89 D8                  mov  rax, r11             (edge count)
    //   0x22  48 83 F8 03               cmp  rax, 3
    //   0x26  72 05                     jb   +5
    //   0x28  B8 03 00 00 00            mov  eax, 3               (clamp)
    //   0x2D  4C 89 C9                  mov  rcx, r9              (plane count)
    //   0x30  48 83 F9 0F               cmp  rcx, 15
    //   0x34  72 05                     jb   +5
    //   0x36  B9 0F 00 00 00            mov  ecx, 15              (clamp)
    //   0x3B  48 8D 04 88               lea  rax, [rax+rcx*4]     (cell index)
    //   0x3F  48 8D 0D BA 08 00 00      lea  rcx, [rip+0x8BA] -> +0x900
    //   0x46  F0 48 FF 04 C1            lock inc qword [rcx+rax*8]
    //   0x4B  59                        pop  rcx
    //   0x4C  58                        pop  rax
    //   0x4D  FF 25 00 00 00 00         jmp  qword ptr [rip+0]
    //   0x53  <qword> continuation
    //   0x60  <qword> counter
    //   0x100 <2048-byte portal bit-set>
    //   0x900 <64-qword (r11,r9) histogram>
    // WIDENED 2026-08-05. The first version masked to 0xFFF (4096 bits), and the
    // very first capture reported area 180 as the most frequent degenTo target
    // on a map where dword_181748D8C = 179, i.e. OUT OF RANGE. That is either an
    // engine sentinel — 0x2EC8A8 does `cmp r10d, dword_181748D8C / jnb`, so
    // out-of-range neighbours genuinely occur — or the mask wrapping a portal
    // index above 4095 onto the wrong table entry. Those are not distinguishable
    // from the data, so the table is widened past any plausible portal count and
    // the drain now reports the highest bit it actually saw. maxBit comfortably
    // below the limit means no aliasing and the attribution stands; maxBit at or
    // near it means the numbers above were fiction.
    constexpr std::size_t kPortalBitsOffset  = 0x100;
    constexpr std::size_t kPortalBitsBytes   = 2048;     // 16384 bits, matches the mask
    constexpr std::uintptr_t kRvaPortalTable = 0x1748D00;  // qword_181748D00
    // qword_181748CF8 — the PER-AREA portal range table, 8 bytes per area:
    //   +0  word  portal COUNT   (read at 0x2EC92A as the loop bound)
    //   +2  word  portal START   (read at 0x2EB934 as the base index)
    // Portals are therefore contiguous per area, which is what makes a portal
    // index reverse-resolvable to its OWNING area with no extra instrumentation:
    // the recorded bit is var_7B8 = portalIdx*3, so portalIdx = bit/3 and the
    // source is the single area whose [start, start+count) contains it.
    constexpr std::uintptr_t kRvaAreaPortalRange = 0x1748CF8;

    // =======================================================================
    // [DegenPair] — the JOINT (r11, r9) distribution at each degen reject.
    //
    // WHY THIS AND NOT ANOTHER COUNTER. Read from the disassembly 2026-08-05:
    // the two counts are not interchangeable, and only one of them is lethal.
    //
    //   0x2EC671  cmp r11,3   -> r15d -> ecx -> a1 -> rec[+0]  EDGE count
    //   0x2EC67B  cmp r9,3    -> esi  -> edx -> a2 -> rec[+2]  PLANE count
    //   (0x2EC6A2 mov r15d,r11d / 0x2EC6AC mov esi,r9d / 0x2EC6F5-FA call E7C70)
    //
    // sub_1802ED900 then consumes that record. Its two count-driven loops are
    // NOT guarded alike:
    //
    //   rec[+0] == 0  ->  0x1802EDA84 `test rax,rax / jz 0x1802EDD39` SKIPS the
    //                     edge loop entirely. Safe. sub_1802E8A20 is safe too:
    //                     its copy count is ((4*A+23)&~0xF)>>4, which floors
    //                     at 1 even for A == 0.
    //   rec[+2] == 0  ->  0x1802EDA30-0x1802EDA74 is a POST-TEST loop
    //                     (rdx=0; inc rdx; cmp rdx,r11; jnz), so a bound of
    //                     zero never terminates and it writes 16 bytes per
    //                     iteration off the end of unk_181E60EF0. 0x1802EDA45
    //                     is `mov [r10+rcx-18h], eax` — exactly the reported
    //                     faulting instruction, and exactly a write.
    //
    // So site 14 as written (BOTH cmps relaxed to 0) can only have crashed via
    // the r9 half. The r11 half is safe on its own — but it is a NO-OP if r9
    // is also below 3 whenever r11 is, because the untouched `cmp r9,3` two
    // instructions later would reject the portal anyway.
    //
    // That is the whole question this probe answers, and it could not be
    // answered from what was already measured: g_clipDegenB counts the branch
    // at 0x2EC67F, which is only REACHED once the r11 gate has passed. It read
    // 0 on every frame of every capture, but that means "given >=3 edges,
    // planes are never <3" — it says nothing about the planes when the edges
    // are degenerate, which is the only case that matters here. Measuring one
    // gate from behind another gate is the same mistake as handoff v2 §6.1/6.3.
    //
    // LAYOUT: cell = r9bucket*4 + r11bucket, r11 clamped to 0..3, r9 to 0..15.
    // Both are RAW values, not thresholds — bucket 15 is the only lumped one,
    // and it exists solely because the table has to be finite.
    //
    // BUILT-IN SELF-CHECK: r11bucket == 3 must read ZERO. The stub only runs on
    // the TAKEN edge of `cmp r11,3 / jb`, and `jb` is unsigned, so every entry
    // has r11 in {0,1,2} by construction. A non-zero 3-bucket means the stub is
    // reading the wrong register and every number on the line is fiction —
    // the same verify-the-probe check that caught the degenTo aliasing.
    //
    // Only site A gets this. Site B (0x2EC67F) stays a plain counter: it sits
    // BEHIND the A gate, so with the code unpatched it can only ever see the
    // r11 >= 3 population, which is not the population in question. If the
    // r11-only salvage is ever applied, B becomes the interesting site and
    // should be moved onto this same recorder then.
    constexpr std::size_t kDegenHistOffset = 0x900;      // past the bit table
    constexpr std::size_t kDegenHistCells  = 64;         // 16 r9 buckets x 4 r11
    constexpr std::size_t kDegenHistBytes  = kDegenHistCells * sizeof(std::uint64_t);
    constexpr std::size_t kDegenR11Max     = 3;          // clamp, inclusive
    constexpr std::size_t kDegenR9Max      = 15;         // clamp, inclusive
    // Where the recorder's own call counter lives. Was 0x30 while the stub was
    // 0x2D of code; the pair histogram grew it to 0x5B, so 0x30 now falls
    // inside an instruction. Named, asserted against the emitted length, and
    // still comfortably below the bit table at 0x100.
    constexpr std::size_t kStubCounterOffset = 0x60;
    static_assert(kStubCounterOffset + sizeof(std::uint64_t) <= kPortalBitsOffset,
                  "counter overlaps the portal bit-set");
    static_assert(kPortalBitsOffset + kPortalBitsBytes <= kDegenHistOffset,
                  "histogram overlaps the portal bit-set");
    static_assert(kDegenHistOffset + kDegenHistBytes <= 4096,
                  "stub page is one 4096-byte AllocateNearPage allocation");

    std::uint8_t* s_clipDegenAPage = nullptr;

    bool InstallBranchPortalRecorder(const char* tag, std::uintptr_t rva,
                                     std::size_t branchSize, std::size_t dispOffset,
                                     const std::uint8_t* expect,
                                     std::uintptr_t origTargetRva,
                                     std::int32_t rbpDisp,
                                     volatile std::uint64_t** counterOut,
                                     std::uint8_t** pageOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* base   = reinterpret_cast<std::uint8_t*>(client);
      auto* target = base + rva;

      if (std::memcmp(target, expect, branchSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] byte mismatch at client.dll+0x", std::hex, rva,
          std::dec, " — not this build, recorder skipped"));
        return false;
      }
      std::int32_t curDisp = 0;
      std::memcpy(&curDisp, expect + dispOffset, sizeof(curDisp));
      if (rva + branchSize + curDisp != origTargetRva) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] branch destination is not 0x", std::hex, origTargetRva,
          std::dec, " — recorder skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(base);
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, kStubCounterOffset + sizeof(std::uint64_t));
      std::memset(page + kPortalBitsOffset, 0, kPortalBitsBytes);
      std::memset(page + kDegenHistOffset, 0, kDegenHistBytes);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      const std::int32_t lockDisp =
        static_cast<std::int32_t>(kStubCounterOffset) - static_cast<std::int32_t>(o + 4);
      std::memcpy(page + o, &lockDisp, 4); o += 4;                    // 0x08
      page[o++] = 0x50;                                               // 0x08 push rax
      page[o++] = 0x51;                                               // 0x09 push rcx
      page[o++] = 0x48; page[o++] = 0x8B; page[o++] = 0x85;           // mov rax,[rbp+disp32]
      std::memcpy(page + o, &rbpDisp, 4); o += 4;                     // 0x11
      page[o++] = 0x48; page[o++] = 0x25;                             // and rax, imm32
      const std::int32_t mask =
        static_cast<std::int32_t>(kPortalBitsBytes * 8u - 1u);        // 0x3FFF
      std::memcpy(page + o, &mask, 4); o += 4;                        // 0x17
      page[o++] = 0x48; page[o++] = 0x0F; page[o++] = 0xAB; page[o++] = 0x05;
      const std::int32_t btsDisp =
        static_cast<std::int32_t>(kPortalBitsOffset) - static_cast<std::int32_t>(o + 4);
      std::memcpy(page + o, &btsDisp, 4); o += 4;                     // 0x1F

      // --- [DegenPair] rax = min(r11, 3) --------------------------------
      // r11 and r9 are the host function's live registers and must not be
      // touched, so both counts are copied into the two pushed scratches
      // before being clamped. `jb` is the unsigned form, matching the site's
      // own `jb`, so a wild count clamps to the top bucket instead of
      // indexing off the table.
      page[o++] = 0x4C; page[o++] = 0x89; page[o++] = 0xD8;           // mov rax, r11
      page[o++] = 0x48; page[o++] = 0x83; page[o++] = 0xF8;
      page[o++] = static_cast<std::uint8_t>(kDegenR11Max);            // cmp rax, 3
      page[o++] = 0x72; page[o++] = 0x05;                             // jb  +5
      page[o++] = 0xB8;                                               // mov eax, imm32
      { const std::int32_t v = static_cast<std::int32_t>(kDegenR11Max);
        std::memcpy(page + o, &v, 4); o += 4; }                       // 0x2D
      // --- rcx = min(r9, 15) --------------------------------------------
      page[o++] = 0x4C; page[o++] = 0x89; page[o++] = 0xC9;           // mov rcx, r9
      page[o++] = 0x48; page[o++] = 0x83; page[o++] = 0xF9;
      page[o++] = static_cast<std::uint8_t>(kDegenR9Max);             // cmp rcx, 15
      page[o++] = 0x72; page[o++] = 0x05;                             // jb  +5
      page[o++] = 0xB9;                                               // mov ecx, imm32
      { const std::int32_t v = static_cast<std::int32_t>(kDegenR9Max);
        std::memcpy(page + o, &v, 4); o += 4; }                       // 0x3B
      // rax = r9bucket*4 + r11bucket. Scale 4 because r11 needs two bits;
      // x86 has no scale-16, which is why the pair is packed this way round.
      page[o++] = 0x48; page[o++] = 0x8D; page[o++] = 0x04;
      page[o++] = 0x88;                                               // lea rax,[rax+rcx*4]
      page[o++] = 0x48; page[o++] = 0x8D; page[o++] = 0x0D;           // lea rcx,[rip+d]
      const std::int32_t histDisp =
        static_cast<std::int32_t>(kDegenHistOffset) - static_cast<std::int32_t>(o + 4);
      std::memcpy(page + o, &histDisp, 4); o += 4;                    // 0x46
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF;
      page[o++] = 0x04; page[o++] = 0xC1;                             // lock inc [rcx+rax*8]

      page[o++] = 0x59;                                               // pop rcx
      page[o++] = 0x58;                                               // pop rax
      page[o++] = 0xFF; page[o++] = 0x25;
      const std::int32_t zero = 0;
      std::memcpy(page + o, &zero, 4); o += 4;                        // 0x53
      const auto contAddr = reinterpret_cast<std::uint64_t>(base + origTargetRva);
      std::memcpy(page + o, &contAddr, sizeof(contAddr)); o += 8;     // 0x5B
      // The counter has to clear the code, and the code grew from 0x2D to 0x5B
      // when the pair histogram went in — the old hard-coded 0x30 now lands in
      // the middle of `cmp rcx, 15`. Checked rather than assumed: this runs
      // before the branch is retargeted, so a stub that outgrew its own page
      // layout declines to install instead of scribbling on its counter.
      if (o > kStubCounterOffset) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub code is ", o, " bytes, counter sits at 0x",
          std::hex, kStubCounterOffset, std::dec, " — recorder skipped"));
        return false;
      }
      const std::uint64_t zero64 = 0;
      std::memcpy(page + kStubCounterOffset, &zero64, sizeof(zero64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) +
         static_cast<std::int64_t>(branchSize));
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub out of rel32 range, recorder skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target + dispOffset, 4, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(target + dispOffset, &rel32, sizeof(rel32));
      DWORD tmp = 0;
      VirtualProtect(target + dispOffset, 4, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, branchSize);

      *counterOut =
        reinterpret_cast<volatile std::uint64_t*>(page + kStubCounterOffset);
      *pageOut    = page;

      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] RECORDER INSTALLED at client.dll+0x", std::hex, rva,
        " stub=0x", reinterpret_cast<std::uintptr_t>(page),
        " -> client.dll+0x", origTargetRva, " rbp+0x", rbpDisp, std::dec,
        " (branch retarget + portal bit-set + (r11,r9) histogram)"));
      return true;
    }

    constexpr std::uintptr_t kRvaEd900 = 0x2ED900;   // sub_1802ED900
    constexpr std::uintptr_t kRvaEb620 = 0x2EB620;   // sub_1802EB620
    // [Ed900Drop] — loc_1802EBE71, the ONLY target of the `jz` at 0x2EB8D0,
    // i.e. sub_1802ED900 returned -1 and the area is dropped. Verified by xref:
    // exactly one code reference, 0x1802EB8D0, so this counter cannot merge
    // anything else. Continues at loc_1802EC957 (the shared loop tail, whose
    // other predecessor is 0x2EC950).
    constexpr std::uintptr_t kRvaEd900Drop     = 0x2EBE71;
    constexpr std::uintptr_t kRvaEd900DropCont = 0x2EC957;
    volatile std::uint64_t*  s_ed900Counter = nullptr;
    volatile std::uint64_t*  s_eb620Counter = nullptr;
    volatile std::uint64_t*  s_ed900DropCounter = nullptr;

    // [ClipDegen] — the two "clipped polygon has fewer than 3 vertices, abandon
    // the portal" rejects at the end of the clip body. These are the ONLY
    // rejects left unpatched on the area-portal path, and they are the leading
    // suspects for the residual PITCH dependence:
    //   yaw   a20 across bins 9.4 - 27.5, no collapse  (sites 12+13 fixed it)
    //   pitch a20 32.67 at +10deg -> 6.42 at -50deg, monotonic, allocFail = 0
    // with [AreaSeed] listLen pinned at 30 (min == max) in EVERY pitch bin, so
    // the candidate set is not the cause on this axis either.
    //
    // Sites 12 and 13 both neutralise predicates that treat all four frustum
    // side planes symmetrically, and a symmetric predicate cannot produce an
    // axis-asymmetric result — so the residual has to be a different one. These
    // two fire only when a portal genuinely STRADDLES and the real clipper at
    // 0x2EBCF1 runs, and that clipper clips against xmmword_1811FC030, the
    // camera FORWARD. Grazing geometry against floor- and ceiling-adjacent
    // portal edges is exactly what pitching produces.
    //
    // COUNTED, NOT PATCHED, deliberately. Neither has a clean accept target:
    // 0x2EC685-0x2EC6B2 is the successful-clip fixup that publishes the new
    // counts (r15d = r11d, esi = r9d) and restores rbx/r13/rdi/r12, so jumping
    // over it would leave the clipper's register state live. Measure first.
    constexpr std::uintptr_t kRvaClipDegenA   = 0x2EC675;  // cmp r11,3 / jb
    constexpr std::uintptr_t kRvaClipDegenB   = 0x2EC67F;  // cmp r9,3  / jb
    constexpr std::uintptr_t kRvaClipDegenTgt = 0x2EC8ED;  // both go here
    volatile std::uint64_t*  s_clipDegenACounter = nullptr;
    volatile std::uint64_t*  s_clipDegenBCounter = nullptr;

    // =======================================================================
    // [QueueProbe] — THE QUEUE CURSOR AND THE SELECTOR-LESS SKIP.
    //
    // WHY. The POSITION sweep (2026-08-05, view locked at pitch -7.81836 /
    // yaw 175.265, 365 frames, only camPos moving) found a20 stepping 14 -> 9
    // across a single sharp plane at y ~ -9984, taking m1 3719 -> 1859 and
    // inst ~10450 -> ~5790 with it. Areas 113, 141, 148, 149, 153 drop as a
    // block (110/110 frames present above, 0/202 below) and 124 <-> 125 swap.
    // Every measured gate is CONSTANT across that step: [AreaSeed] listLen=30,
    // ed900Drop=4.00, allocFail=0, clipDegen=0/0. So nothing that is currently
    // instrumented is doing it, and the five areas are not being REJECTED --
    // their portal crossing is never attempted.
    //
    // WHAT THE DISASSEMBLY THEN SHOWED, and it is a quantity nobody has ever
    // logged. sub_1802EAD60 does not only FILL the order list, it RETURNS the
    // index the queue loop starts at:
    //     1802eb7e1  call sub_1802EAD60
    //     1802eb7f7  mov  ebx, eax                          <- ebx = START INDEX
    //     1802eb860  mov  eax, ebx / inc ebx                <- queue loop head
    //     1802eb864  movzx ecx, word_1811FE920[rsi+rax*2]   <- walks FORWARD
    // The loop only ever walks forward from that cursor, so every order-list
    // slot BELOW it is unreachable no matter what any portal accepts.
    // [AreaSeed] measured listLen (how many entries EAD60 wrote) and found it
    // pinned at 30 -- which is why EAD60 was recorded as "eliminated". But
    // listLen flat does NOT imply the RETURN VALUE is flat, and the return
    // value is the position-keyed one. Same class of error as CULLING_BIBLE's
    // withdrawn claim #5, one level further down: an exclusion is only valid
    // for the QUANTITY it was measured on, not just the axis.
    //
    // TWO CAPTURES, because they answer different halves:
    //   qStart  EAD60's return value. Falls with position => the cursor is the
    //           gate and the five areas are simply never visited.
    //   qSkip   areas VISITED by the loop but skipped at 0x2EB884 because their
    //           selector is still -1, i.e. no portal crossing ever enqueued
    //           them. Flat qStart + these five in qSkip => the cursor is
    //           innocent and the loss is in the portal crossings after all.
    // The two are mutually exclusive explanations and one line separates them.
    //
    // Both are hand-built asm islands, NOT C wrappers. sub_1802EB620 does not
    // decompile, and the [DispProbe] freeze of 2026-08-05 was caused by
    // wrapping a non-decompiling SIMD function with a guessed signature. An
    // island commits to no calling convention: it touches memory it owns, one
    // scratch register it pushes, and flags that are provably dead.
    constexpr std::uintptr_t kRvaQStart     = 0x2EB7EF;  // or r8,-1 / xor edx,edx
    constexpr std::uintptr_t kRvaQStartCont = 0x2EB7F5;  // not cl
    // The `jz loc_1802EC95E` taken when an order-list area has selector == -1.
    // Retargeted (condition preserved), so the island runs ONLY on the skip.
    constexpr std::uintptr_t kRvaQSkip      = 0x2EB884;
    constexpr std::uintptr_t kRvaQSkipTgt   = 0x2EC95E;
    // Island slots. 0x30 counter (shared convention), 0x38 latched value,
    // 0x40 area bitmap. 256 bits covers dword_181748D8C = 179 areas with room.
    constexpr std::size_t    kQValueOffset  = 0x38;
    constexpr std::size_t    kQAreaBitsOff  = 0x40;
    constexpr std::size_t    kQAreaBitsSize = 32;   // 256 bits
    volatile std::uint64_t*  s_qStartCounter = nullptr;
    volatile std::uint64_t*  s_qSkipCounter  = nullptr;
    std::uint8_t*            s_qStartPage    = nullptr;
    std::uint8_t*            s_qSkipPage     = nullptr;

    // =======================================================================
    // [FaceReject] — client.dll+0x2EB98F, THE CAMERA-BEHIND-THE-PORTAL TEST.
    //
    // Measured 2026-08-05: qStart is pinned at 149 (min == max over 259
    // view-locked frames) so sub_1802EAD60's cursor is NOT the gate, while
    // areas 92/113/141/148/153 sit in qSkipAreas on 152 of 156 LOW frames
    // against 4 of 103 HIGH ones. So those areas ARE reached by the queue loop
    // and no portal crossing ever writes them a selector. This is the reject
    // that explains it, and it is the FIRST test in the portal body:
    //
    //   1802eb956  movaps xmm0, [portal plane]         ; float4
    //   1802eb965  mulps  xmm0, xmmword_1811FC000      ; x CAMERA ORIGIN
    //              ...two shufps/addps = horizontal add = plane . camOrigin
    //   1802eb983  mulss  xmm0, dword_180911100[rax*4] ; x side sign
    //   1802eb98c  comiss xmm1(0), xmm0
    //   1802eb98f  jnb    loc_1802EC923                ; dist <= 0 -> SKIP
    //
    // dword_180911100 is a two-entry sign table {+1.0f, -1.0f} (the bytes
    // after it are the string "Draw por..."), so this is a signed plane-side
    // test on the camera ORIGIN -- a pure function of camera POSITION, and the
    // only one of the six rejects on this path that reads position rather than
    // orientation. That is why the whole area layer measured flat across pitch
    // and yaw and steps hard across a plane in y: a plane-side test IS a sharp
    // boundary, and the observed one sits at y ~ -9984.
    //
    // It is upstream of every reject already instrumented -- the outcode test
    // (areaPortal), the clip loop (areaClip/areaSkipClip), the <3 rejects
    // (clipDegen) and the allocator (allocFail) all sit BELOW it -- which is
    // exactly why all of them read zero while areas still vanish. loc_1802EC923
    // falls into the loop tail at 0x2EC92A, so the reject means "next portal",
    // with no allocator call and no selector write.
    //
    // COUNTED, NOT PATCHED. This is legitimate portal logic (you may only see
    // through a portal from its front face), so a patch here is a behaviour
    // change, not a bug fix, and rtx.conf currently says "Do not add a site
    // 16". Measure the identities first.
    //
    // var_7B8 (the portal bit index, portalIdx*3) is written at 0x2EB93F,
    // BEFORE this branch, so the existing portal recorder resolves the target
    // area here exactly as it does at 0x2EC675 -- same rbp displacement, same
    // qword_181748D00 lookup at drain time.
    //
    // !! THE (r11,r9) HISTOGRAM ON THIS PAGE IS MEANINGLESS AND MUST NOT BE
    // !! READ. InstallBranchPortalRecorder always emits it, but r9d and r11d
    // !! are not loaded until 0x2EB995/0x2EB9AB -- AFTER this site -- so at
    // !! 0x2EB98F they hold unrelated leftovers. They are only ever read into
    // !! the stub's own pushed scratch and clamped, so the histogram is safe
    // !! but junk. Only the counter and the portal bitmap mean anything here.
    constexpr std::uintptr_t kRvaFaceReject    = 0x2EB98F;
    constexpr std::uintptr_t kRvaFaceRejectTgt = 0x2EC923;
    volatile std::uint64_t*  s_faceRejectCounter = nullptr;
    std::uint8_t*            s_faceRejectPage    = nullptr;

    // =======================================================================
    // [PortalWalk] — client.dll+0x2EB93F, EVERY portal the flood iterates.
    //
    // WHY, and it is the measurement that should have come first. [FaceReject]
    // refuted 0x2EB98F: areas 113/141/148 appear in faceAreas ZERO times in
    // both states while their qSkip goes 7/91 -> 278/281. An area that is
    // never rejected and never enqueued was never REACHED -- so the portal
    // leading to it is never walked, and the loss is a CASCADE whose root is
    // some other area being dropped first. No reject fires anywhere along the
    // way, which is why six instrumented sites all read zero.
    //
    // This records the target of every portal the loop actually walks, which
    // splits the remaining possibilities cleanly and without another guess:
    //   target in walkAreas but never enqueued => a seventh reject exists on
    //     the crossing path and the enumeration is still incomplete.
    //   target ABSENT from walkAreas           => confirmed cascade; nothing
    //     rejected it, its source area simply never ran its portal loop. The
    //     next question is then which source, not which reject.
    //
    // SITE. 0x2EB93F is `mov [rbp+0F8h], rcx`, the store of var_7B8 at the top
    // of the portal body -- before the facing test, before the outcode test,
    // before the clip. Every portal iteration passes through it exactly once,
    // and rcx already holds portalIdx*3, so the island needs no frame access
    // at all: it bts's rcx directly into the same bit table the other two
    // recorders use, and DrainPortalAreasFromPage resolves it identically.
    //
    // The stolen instruction is base+disp only (no rip-relative), so it
    // replays from the island unchanged.
    //
    // FLAGS ARE DEAD HERE, checked rather than assumed: the last writer is
    // `add eax, ecx` at 0x2EB939 and the next reader is the `jnb` at 0x2EB98F,
    // which consumes the `comiss` at 0x2EB98C -- no branch in between. The
    // island's lock inc / and / bts are therefore unobservable.
    constexpr std::uintptr_t kRvaPortalWalk = 0x2EB93F;
    volatile std::uint64_t*  s_portalWalkCounter = nullptr;
    std::uint8_t*            s_portalWalkPage    = nullptr;

    // =======================================================================
    // [SelWrite] — client.dll+0x2EC739, THE SELECTOR WRITE ITSELF.
    //
    // The one measurement that separates the last two possibilities. Pairs
    // showed 127>149 and 127>124 walked on 109/109 HIGH and 160/160 LOW
    // frames -- the inbound crossings never stop -- while every OUTBOUND edge
    // from 149 and 124 closes (108 -> 16). So those two areas stop being
    // DISPATCHED, not stop being reached. And they are never selector-less
    // either: qSkipAreas contains neither of them in either state, so the
    // queue loop does not visit-and-reject them, it never visits them at all.
    // Every counted reject stays flat across the step (clipDegen 0.00,
    // allocFail 0.00, ed900Drop 4.00) and faceReject on 149 runs BACKWARDS
    // (108 HIGH vs 16 LOW). Those facts cannot all hold at once unless either:
    //
    //   (1) the selector is never written -- a seventh reject between
    //       0x2EB93F and 0x2EC739 that none of the six sites cover; or
    //   (2) the selector IS written and the cursor never comes back for it.
    //       qStart=149 is a SLOT index (its equality with area id 149 is a
    //       coincidence that made this harder to see). The loop walks slots
    //       FORWARD from 149, so an area whose slot is below the cursor is
    //       never visited -- and never reported selector-less, because being
    //       reported requires being visited. In HIGH the rewind at
    //       0x2EC8DA/0x2EC8E2 would be what rescues it.
    //
    // The ordering theory was dismissed earlier BECAUSE that rewind exists.
    // That was an error of the same kind as the rest of this investigation:
    // the rewind's existence was verified, its FIRING was not. dword_1811FBD98
    // is incremented on exactly that path and has never been read.
    //
    // 149 present here in LOW  => (2): enqueued, then abandoned by the cursor.
    // 149 absent here in LOW   => (1): back to the disassembly.
    //
    // SITE. 0x2EC739 is the store itself:
    //   42 89 84 92 1C F9 1F 01   mov [rdx+r10*4+11FF91Ch], eax
    // rdx carries the 0x180000000 image base (lea at 0x2EC71C) and r10 the
    // TARGET AREA, so the instruction is base+index+disp32 with no
    // rip-relative component and replays from an island unchanged. Steal is 8,
    // exactly kIslandMaxSteal.
    //
    // Area ids, not portal bits, so this island uses its own 32-byte table at
    // 0x80 -- clear of the counter at 0x60 and of the portal bit table at
    // 0x100, neither of which this page uses.
    // =======================================================================
    // [DropAreas] — WHICH areas sub_1802ED900 drops, not how many.
    //
    // ed900Drop has read 4.00 per frame, constant, in every capture on both
    // sides of the step, and was treated as exonerated on that basis. That is
    // the same count-for-identity substitution that already cost this
    // investigation listLen (30 flat, contents never compared) and the 16-cap
    // on the seed list. A constant NUMBER of drops says nothing about WHICH
    // areas are dropped, and only the identity matters here.
    //
    // The drop is the one path that fits every measurement at once. 0x2EB8D0
    // jumps here when ED900 returns -1, which is AFTER the selector is
    // consumed at 0x2EB88A but BEFORE the dispatch at 0x2EB910 and the portal
    // loop at 0x2EB915. So a dropped area:
    //   - was enqueued            => [SelWrite] sees it            (observed)
    //   - had its selector eaten  => live=0 at exit                (observed)
    //   - never runs its portals  => no outbound walkPairs edge    (observed)
    //   - is not selector-less    => absent from qSkipAreas        (observed)
    //   - leaves nothing pending  => pendExit=0                    (observed)
    // Nothing else on this path produces all five.
    //
    // And it is consistent with the A/B: with the area patches ON more portals
    // cross, so ED900 builds bigger/more records (alloc 93 vs 40, selN 73 vs
    // 28) and returns -1 for areas it would otherwise accept. Turning the area
    // group off restores 124/141/148/149/153 (ran 2/138 -> 15/17).
    //
    // The area id is read from var_7C0 rather than rbx: rbx is callee-saved and
    // should survive the `call sub_1802ED900` at 0x2EB8C5, but "should" is not
    // a measurement, and the frame slot is written explicitly at 0x2EB87C
    // (`mov [rbp+0F0h], rcx`, displacement confirmed from the encoding) and is
    // live across the whole iteration.
    constexpr std::int32_t   kRbpDispAreaId  = 0xF0;      // var_7C0
    volatile std::uint64_t*  s_dropAreaCounter = nullptr;
    std::uint8_t*            s_dropAreaPage    = nullptr;

    constexpr std::uintptr_t kRvaSelWrite     = 0x2EC739;
    constexpr std::size_t    kSelAreaBitsOff  = 0x80;
    constexpr std::size_t    kSelAreaBitsSize = 32;         // 256 bits
    static_assert(kSelAreaBitsOff >= kStubCounterOffset + sizeof(std::uint64_t),
                  "selector area table overlaps the stub counter");
    static_assert(kSelAreaBitsOff + kSelAreaBitsSize <= kPortalBitsOffset,
                  "selector area table overlaps the portal bit table");
    // dword_1811FBD98 — incremented at 0x2EC8DC on the cursor REWIND path,
    // i.e. every time an enqueued area's slot was behind the cursor and the
    // loop had to go back for it. Read directly; no hook needed.
    constexpr std::uintptr_t kRvaRewindCount  = 0x11FBD98;
    volatile std::uint64_t*  s_selWriteCounter = nullptr;
    std::uint8_t*            s_selWritePage    = nullptr;

    // Steal-and-replay island that bts's RCX (masked) into the portal bit
    // table, for a site that is not a branch.
    //   +0x00  F0 48 FF 05 <d32>     lock inc [rip -> kStubCounterOffset]
    //   +0x08  50                    push rax
    //   +0x09  48 89 C8              mov  rax, rcx
    //   +0x0C  48 25 FF 3F 00 00     and  rax, 3FFFh
    //   +0x12  48 0F AB 05 <d32>     bts  [rip -> kPortalBitsOffset], rax
    //   +0x1A  58                    pop  rax
    //   +0x1B  <stolen>              (position independent)
    //   +0x1B+n FF 25 00 00 00 00    jmp  [rip+0]
    //   +0x21+n <qword> continue
    // Tail-jmp island that ALSO records an area id read from [rbp+disp32].
    // Same shape as InstallCounterIslandTailJmp (steal a block ending in an
    // unconditional `jmp rel32`, replay the head, resolve the jmp to an
    // absolute), with the bts added.
    //   +0x00  F0 48 FF 05 <d32>   lock inc [rip -> kStubCounterOffset]
    //   +0x08  50                  push rax
    //   +0x09  48 8B 85 <d32>      mov  rax, [rbp+disp32]
    //   +0x10  48 25 FF 00 00 00   and  rax, 0FFh
    //   +0x16  48 0F AB 05 <d32>   bts  [rip -> kSelAreaBitsOff], rax
    //   +0x1E  58                  pop  rax
    //   +0x1F  <replayed head>
    //   +....  FF 25 00 00 00 00   jmp  [rip+0]
    //   +....  <qword> resolved destination
    bool InstallTailJmpAreaRecorder(const char* tag, std::uintptr_t rva,
                                    std::size_t stealSize, std::size_t headLen,
                                    const std::uint8_t* expect,
                                    std::uintptr_t contRva, std::int32_t rbpDisp,
                                    volatile std::uint64_t** counterOut,
                                    std::uint8_t** pageOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* base   = reinterpret_cast<std::uint8_t*>(client);
      auto* target = base + rva;

      if (std::memcmp(target, expect, stealSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] byte mismatch at client.dll+0x", std::hex, rva,
          std::dec, " - not this build, recorder skipped"));
        return false;
      }
      // Re-derive the stolen jmp's destination and refuse if it moved, exactly
      // as the plain tail-jmp counter does.
      std::int32_t curDisp = 0;
      std::memcpy(&curDisp, expect + headLen + 1, sizeof(curDisp));
      if (rva + stealSize + curDisp != contRva) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] tail jmp does not land on 0x", std::hex, contRva,
          std::dec, " - recorder skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(base);
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, kStubCounterOffset + sizeof(std::uint64_t));
      std::memset(page + kSelAreaBitsOff, 0, kSelAreaBitsSize);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      { const std::int32_t d = static_cast<std::int32_t>(kStubCounterOffset)
                             - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }
      page[o++] = 0x50;                                       // push rax
      page[o++] = 0x48; page[o++] = 0x8B; page[o++] = 0x85;   // mov rax,[rbp+d32]
      { std::memcpy(page + o, &rbpDisp, 4); o += 4; }
      page[o++] = 0x48; page[o++] = 0x25;                     // and rax, 0FFh
      { const std::int32_t m = 0xFF; std::memcpy(page + o, &m, 4); o += 4; }
      page[o++] = 0x48; page[o++] = 0x0F; page[o++] = 0xAB; page[o++] = 0x05;
      { const std::int32_t d = static_cast<std::int32_t>(kSelAreaBitsOff)
                             - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }
      page[o++] = 0x58;                                       // pop rax
      std::memcpy(page + o, expect, headLen); o += headLen;   // replay head
      page[o++] = 0xFF; page[o++] = 0x25;
      { const std::int32_t z = 0; std::memcpy(page + o, &z, 4); o += 4; }
      const auto contAddr = reinterpret_cast<std::uint64_t>(base + contRva);
      std::memcpy(page + o, &contAddr, sizeof(contAddr)); o += 8;
      if (o > kStubCounterOffset) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub is ", o, " bytes and would overrun its counter"));
        return false;
      }
      const std::uint64_t z64 = 0;
      std::memcpy(page + kStubCounterOffset, &z64, sizeof(z64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] island out of rel32 range, recorder skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, stealSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      std::uint8_t stub[kIslandMaxSteal];
      std::memset(stub, 0x90, sizeof(stub));
      stub[0] = 0xE9;
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, stealSize);
      DWORD tmp = 0;
      VirtualProtect(target, stealSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, stealSize);

      *counterOut = reinterpret_cast<volatile std::uint64_t*>(
        page + kStubCounterOffset);
      *pageOut = page;
      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] RECORDER INSTALLED at client.dll+0x", std::hex, rva,
        " island=0x", reinterpret_cast<std::uintptr_t>(page),
        " -> client.dll+0x", contRva, " rbp+0x", rbpDisp, std::dec,
        " (tail-jmp + dropped-area bitmap)"));
      return true;
    }

    // `movRax` is the 3-byte `mov rax, <src>` that copies the register holding
    // the value to record (48 89 C8 = rcx, 4C 89 D0 = r10). `mask` bounds it so
    // the bts cannot address outside the table, and (bitsOff, bitsBytes) select
    // which table on the page it lands in — portal bits at 0x100 for a portal
    // index, or a 32-byte area table for a raw area id.
    bool InstallRegBitIsland(const char* tag, std::uintptr_t rva,
                             std::size_t stealSize, const std::uint8_t* expect,
                             const std::uint8_t* movRax, std::int32_t mask,
                             std::size_t bitsOff, std::size_t bitsBytes,
                             volatile std::uint64_t** counterOut,
                             std::uint8_t** pageOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* base   = reinterpret_cast<std::uint8_t*>(client);
      auto* target = base + rva;

      if (std::memcmp(target, expect, stealSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] byte mismatch at client.dll+0x", std::hex, rva,
          std::dec, " - not this build, probe skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(base);
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, kStubCounterOffset + sizeof(std::uint64_t));
      std::memset(page + bitsOff, 0, bitsBytes);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      { const std::int32_t d = static_cast<std::int32_t>(kStubCounterOffset)
                             - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }
      page[o++] = 0x50;                                       // push rax
      page[o++] = movRax[0]; page[o++] = movRax[1]; page[o++] = movRax[2];
      page[o++] = 0x48; page[o++] = 0x25;                     // and rax, imm32
      { std::memcpy(page + o, &mask, 4); o += 4; }
      page[o++] = 0x48; page[o++] = 0x0F; page[o++] = 0xAB; page[o++] = 0x05;
      { const std::int32_t d = static_cast<std::int32_t>(bitsOff)
                             - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }
      page[o++] = 0x58;                                       // pop rax
      std::memcpy(page + o, expect, stealSize); o += stealSize;
      page[o++] = 0xFF; page[o++] = 0x25;
      { const std::int32_t z = 0; std::memcpy(page + o, &z, 4); o += 4; }
      const auto contAddr = reinterpret_cast<std::uint64_t>(target + stealSize);
      std::memcpy(page + o, &contAddr, sizeof(contAddr)); o += 8;
      if (o > kStubCounterOffset) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub is ", o, " bytes, counter sits at 0x",
          std::hex, kStubCounterOffset, std::dec, " - probe skipped"));
        return false;
      }
      const std::uint64_t z64 = 0;
      std::memcpy(page + kStubCounterOffset, &z64, sizeof(z64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] island out of rel32 range, probe skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, stealSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      std::uint8_t stub[kIslandMaxSteal];
      std::memset(stub, 0x90, sizeof(stub));
      stub[0] = 0xE9;
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, stealSize);
      DWORD tmp = 0;
      VirtualProtect(target, stealSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, stealSize);

      *counterOut = reinterpret_cast<volatile std::uint64_t*>(
        page + kStubCounterOffset);
      *pageOut = page;
      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] INSTALLED at client.dll+0x", std::hex, rva,
        " island=0x", reinterpret_cast<std::uintptr_t>(page), std::dec,
        " (steal+replay, bts rcx = portal bit, every walked portal)"));
      return true;
    }

    // Steal-and-replay island that LATCHES EAX into the island page.
    //   +0x00  F0 48 FF 05 <d32>   lock inc [rip -> +0x30]
    //   +0x08  89 05 <d32>         mov [rip -> +0x38], eax
    //   +0x0E  <stolen bytes>
    //   +0x0E+n FF 25 00 00 00 00  jmp [rip+0]
    //   +0x14+n <qword> continue address
    // Flags: the lock inc writes them, but both stolen instructions here write
    // flags themselves afterwards, so nothing downstream can observe mine.
    // Registers: EAX is read, never written. No push, no stack use at all.
    bool InstallEaxLatchIsland(const char* tag, std::uintptr_t rva,
                               std::size_t stealSize, const std::uint8_t* expect,
                               volatile std::uint64_t** counterOut,
                               std::uint8_t** pageOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* base   = reinterpret_cast<std::uint8_t*>(client);
      auto* target = base + rva;

      if (std::memcmp(target, expect, stealSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] byte mismatch at client.dll+0x", std::hex, rva,
          std::dec, " - not this build, probe skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(base);
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, 0x48);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      { const std::int32_t d = 0x30 - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }                  // o == 0x08
      page[o++] = 0x89; page[o++] = 0x05;
      { const std::int32_t d = static_cast<std::int32_t>(kQValueOffset)
                             - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }                  // o == 0x0E
      std::memcpy(page + o, expect, stealSize); o += stealSize;
      page[o++] = 0xFF; page[o++] = 0x25;
      { const std::int32_t z = 0; std::memcpy(page + o, &z, 4); o += 4; }
      const auto contAddr = reinterpret_cast<std::uint64_t>(target + stealSize);
      std::memcpy(page + o, &contAddr, sizeof(contAddr)); o += 8;
      if (o > 0x30) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub is ", o, " bytes and would overrun its counter"));
        return false;
      }
      const std::uint64_t z64 = 0;
      std::memcpy(page + 0x30, &z64, sizeof(z64));
      std::memcpy(page + kQValueOffset, &z64, sizeof(z64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) + 5);
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] island out of rel32 range, probe skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target, stealSize, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      std::uint8_t stub[kIslandMaxSteal];
      std::memset(stub, 0x90, sizeof(stub));
      stub[0] = 0xE9;
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(stub + 1, &rel32, sizeof(rel32));
      std::memcpy(target, stub, stealSize);
      DWORD tmp = 0;
      VirtualProtect(target, stealSize, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, stealSize);

      *counterOut = reinterpret_cast<volatile std::uint64_t*>(page + 0x30);
      *pageOut    = page;
      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] INSTALLED at client.dll+0x", std::hex, rva,
        " island=0x", reinterpret_cast<std::uintptr_t>(page), std::dec,
        " (latches eax = sub_1802EAD60's queue start index)"));
      return true;
    }

    // Branch recorder that sets one bit per AREA ID held in ECX. Same retarget
    // trick as InstallBranchPortalRecorder: only the rel32 displacement is
    // rewritten, so the condition is preserved and the island runs on the taken
    // path only -- where flags are already dead.
    //   +0x00  F0 48 FF 05 <d32>      lock inc [rip -> +0x30]
    //   +0x08  50                     push rax
    //   +0x09  48 89 C8               mov  rax, rcx
    //   +0x0C  48 25 FF 00 00 00      and  rax, 0xFF     (bounds the bts)
    //   +0x12  48 0F AB 05 <d32>      bts  [rip -> +0x40], rax
    //   +0x1A  58                     pop  rax
    //   +0x1B  FF 25 00 00 00 00      jmp  [rip+0]
    //   +0x21  <qword> original branch target
    bool InstallBranchAreaRecorder(const char* tag, std::uintptr_t rva,
                                   std::size_t branchSize, std::size_t dispOffset,
                                   const std::uint8_t* expect,
                                   std::uintptr_t origTargetRva,
                                   volatile std::uint64_t** counterOut,
                                   std::uint8_t** pageOut) {
      HMODULE client = GetModuleHandleA("client.dll");
      if (client == nullptr)
        return false;
      auto* base   = reinterpret_cast<std::uint8_t*>(client);
      auto* target = base + rva;

      if (std::memcmp(target, expect, branchSize) != 0) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] byte mismatch at client.dll+0x", std::hex, rva,
          std::dec, " - not this build, recorder skipped"));
        return false;
      }
      std::int32_t curDisp = 0;
      std::memcpy(&curDisp, expect + dispOffset, sizeof(curDisp));
      if (rva + branchSize + curDisp != origTargetRva) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] branch destination is not 0x", std::hex, origTargetRva,
          std::dec, " - recorder skipped"));
        return false;
      }

      std::uint8_t* page = AllocateNearPage(base);
      if (page == nullptr) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] no page within rel32 range of client.dll"));
        return false;
      }
      std::memset(page, 0xCC, kQAreaBitsOff + kQAreaBitsSize);
      std::memset(page + kQAreaBitsOff, 0, kQAreaBitsSize);

      std::size_t o = 0;
      page[o++] = 0xF0; page[o++] = 0x48; page[o++] = 0xFF; page[o++] = 0x05;
      { const std::int32_t d = 0x30 - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }                   // 0x08
      page[o++] = 0x50;                                           // push rax
      page[o++] = 0x48; page[o++] = 0x89; page[o++] = 0xC8;       // mov rax, rcx
      page[o++] = 0x48; page[o++] = 0x25;
      { const std::int32_t m = 0xFF; std::memcpy(page + o, &m, 4); o += 4; }
      page[o++] = 0x48; page[o++] = 0x0F; page[o++] = 0xAB; page[o++] = 0x05;
      { const std::int32_t d = static_cast<std::int32_t>(kQAreaBitsOff)
                             - static_cast<std::int32_t>(o + 4);
        std::memcpy(page + o, &d, 4); o += 4; }                   // 0x1A
      page[o++] = 0x58;                                           // pop rax
      page[o++] = 0xFF; page[o++] = 0x25;
      { const std::int32_t z = 0; std::memcpy(page + o, &z, 4); o += 4; }
      const auto contAddr = reinterpret_cast<std::uint64_t>(base + origTargetRva);
      std::memcpy(page + o, &contAddr, sizeof(contAddr)); o += 8;
      if (o > 0x30) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub is ", o, " bytes and would overrun its counter"));
        return false;
      }
      const std::uint64_t z64 = 0;
      std::memcpy(page + 0x30, &z64, sizeof(z64));

      const std::int64_t rel =
        static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(page)) -
        (static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(target)) +
         static_cast<std::int64_t>(branchSize));
      if (rel > INT32_MAX || rel < INT32_MIN) {
        dxvk::Logger::warn(dxvk::str::format(
          "[", tag, "] stub out of rel32 range, recorder skipped"));
        return false;
      }

      DWORD oldProt = 0;
      if (!VirtualProtect(target + dispOffset, 4, PAGE_EXECUTE_READWRITE, &oldProt)) {
        dxvk::Logger::warn(dxvk::str::format("[", tag, "] VirtualProtect failed"));
        return false;
      }
      const std::int32_t rel32 = static_cast<std::int32_t>(rel);
      std::memcpy(target + dispOffset, &rel32, sizeof(rel32));
      DWORD tmp = 0;
      VirtualProtect(target + dispOffset, 4, oldProt, &tmp);
      FlushInstructionCache(GetCurrentProcess(), target, branchSize);

      *counterOut = reinterpret_cast<volatile std::uint64_t*>(page + 0x30);
      *pageOut    = page;
      dxvk::Logger::warn(dxvk::str::format(
        "[", tag, "] RECORDER INSTALLED at client.dll+0x", std::hex, rva,
        " stub=0x", reinterpret_cast<std::uintptr_t>(page),
        " -> client.dll+0x", origTargetRva, std::dec,
        " (branch retarget + selector-less area bitmap from ecx)"));
      return true;
    }

    bool DoInstallEd900Probe() {
      // sub_1802ED900:  mov rax,rsp / mov [rax+10h],rbx     = 3 + 4 = 7
      static constexpr std::uint8_t kEd900[7] = {
        0x48, 0x8B, 0xC4, 0x48, 0x89, 0x58, 0x10,
      };
      // sub_1802EB620:  mov rax,rsp / mov [rax+20h],rbx     = 3 + 4 = 7
      static constexpr std::uint8_t kEb620[7] = {
        0x48, 0x8B, 0xC4, 0x48, 0x89, 0x58, 0x20,
      };
      static_assert(sizeof(kEd900) <= kIslandMaxSteal, "stub buffer");
      static_assert(sizeof(kEb620) <= kIslandMaxSteal, "stub buffer");

      bool any = false;
      any |= InstallCounterIsland("Ed900Probe", kRvaEd900, sizeof(kEd900),
                                  kEd900, &s_ed900Counter);
      // [Eb620Probe]: how many times does the AREA BUILDER run per frame?
      // a20 is summed over every EB620 invocation, so a20 alone cannot tell
      // "fewer views doing work" from "fewer areas per view". drains=4 counts
      // sub_1802F04F0, NOT EB620, so the invocation count has never actually
      // been measured. areas/view = a20 / eb620.
      any |= InstallCounterIsland("Eb620Probe", kRvaEb620, sizeof(kEb620),
                                  kEb620, &s_eb620Counter);
      // [Ed900Drop]: how many areas does ED900's -1 DROP, not how many times
      // is ED900 called. Those are different questions and only the second has
      // ever been measured. ed900 sat flat at ~1.0/frame across yaw while a20
      // fell 5 -> 2, and that was read as an acquittal — but a drop at 0x2EB8D0
      // happens BEFORE the dispatch at 0x2EB910 and before the portal loop at
      // 0x2EB915, so one -1 on an area whose portals would have opened the rest
      // takes the entire downstream flood with it. A flat call count is exactly
      // what a single early drop looks like. ED900 also reads xmmword_1811FC030
      // (the camera FORWARD) at four sites, so its VERDICT is view-dependent
      // even where its call count is not.
      //
      //   loc_1802EBE71:  33 D2              xor edx, edx      <- replayed
      //                   E9 DF 0A 00 00     jmp loc_1802EC957 <- resolved
      // Steal 7 (both instructions, so the 5-byte hook jmp fits), replay 2.
      static constexpr std::uint8_t kEd900Drop[7] = {
        0x33, 0xD2,                                       // xor edx, edx
        0xE9, 0xDF, 0x0A, 0x00, 0x00,                     // jmp loc_1802EC957
      };
      static_assert(sizeof(kEd900Drop) <= kIslandMaxSteal, "stub buffer");
      // [DropAreas] replaces the plain Ed900Drop counter with the same island
      // plus the dropped area's IDENTITY. The count alone has read 4.00 flat
      // in every capture and was taken as an acquittal; which four areas it is
      // has never been recorded, and that is the whole question.
      any |= InstallTailJmpAreaRecorder("DropAreas", kRvaEd900Drop,
                                        sizeof(kEd900Drop), 2, kEd900Drop,
                                        kRvaEd900DropCont, kRbpDispAreaId,
                                        &s_dropAreaCounter, &s_dropAreaPage);
      s_ed900DropCounter = s_dropAreaCounter;   // keep ed900Drop= on the line

      // [ClipDegen] A and B. Both are `0F 82 <rel32>` -> loc_1802EC8ED; the
      // installer re-derives the encoded destination from these bytes and
      // refuses if it is not 0x2EC8ED, so a build whose branch moved cannot be
      // miscounted.
      static constexpr std::uint8_t kClipDegenA[6] = {
        0x0F, 0x82, 0x72, 0x02, 0x00, 0x00,               // jb loc_1802EC8ED
      };
      static constexpr std::uint8_t kClipDegenB[6] = {
        0x0F, 0x82, 0x68, 0x02, 0x00, 0x00,               // jb loc_1802EC8ED
      };
      // A gets the recorder (it is the one that fires); B stays a plain counter
      // because it has never incremented once across every capture so far.
      any |= InstallBranchPortalRecorder("ClipDegenA", kRvaClipDegenA, 6, 2,
                                         kClipDegenA, kRvaClipDegenTgt,
                                         0xF8, &s_clipDegenACounter,
                                         &s_clipDegenAPage);
      any |= InstallBranchCounter("ClipDegenB", kRvaClipDegenB, 6, 2,
                                  kClipDegenB, kRvaClipDegenTgt,
                                  &s_clipDegenBCounter);

      // [QueueProbe]. qStart latches sub_1802EAD60's return value -- the index
      // the queue loop starts at, which has never been logged and is the one
      // position-keyed quantity on this path. Steal is the two instructions at
      // 0x2EB7EF, both position independent, and EAX (the return value) is
      // still live there; it is not consumed until `mov ebx, eax` at 0x2EB7F7.
      //   1802eb7ef  49 83 C8 FF   or  r8, -1
      //   1802eb7f3  33 D2         xor edx, edx
      static constexpr std::uint8_t kQStart[6] = {
        0x49, 0x83, 0xC8, 0xFF,                           // or  r8, 0FFFFFFFFFFFFFFFFh
        0x33, 0xD2,                                       // xor edx, edx
      };
      static_assert(sizeof(kQStart) <= kIslandMaxSteal, "stub buffer");
      any |= InstallEaxLatchIsland("QStart", kRvaQStart, sizeof(kQStart),
                                   kQStart, &s_qStartCounter, &s_qStartPage);

      // qSkip records WHICH areas the loop visited and skipped for want of a
      // selector. ECX holds the area id there (loaded at 0x2EB864, still live
      // through the `cmp r9d,-1` at 0x2EB880).
      //   1802eb884  0F 84 D4 10 00 00   jz loc_1802EC95E
      static constexpr std::uint8_t kQSkip[6] = {
        0x0F, 0x84, 0xD4, 0x10, 0x00, 0x00,               // jz loc_1802EC95E
      };
      any |= InstallBranchAreaRecorder("QSkip", kRvaQSkip, 6, 2, kQSkip,
                                       kRvaQSkipTgt, &s_qSkipCounter,
                                       &s_qSkipPage);

      // [FaceReject]. Same recorder as ClipDegenA and the same rbp
      // displacement, because it resolves the target area from the same
      // var_7B8 portal bit index -- verified from the encoding at 0x2EC708,
      // `48 8B 8D F8 00 00 00` = mov rcx,[rbp+0F8h]. var_7B8 is written at
      // 0x2EB93F, before this branch, so it is already the current portal's.
      //   1802eb98f  0F 83 8E 0F 00 00   jnb loc_1802EC923
      // (0x2EB98F + 6 + 0xF8E == 0x2EC923; the installer re-derives that and
      // declines if the branch ever moves.)
      static constexpr std::uint8_t kFaceReject[6] = {
        0x0F, 0x83, 0x8E, 0x0F, 0x00, 0x00,               // jnb loc_1802EC923
      };
      any |= InstallBranchPortalRecorder("FaceReject", kRvaFaceReject, 6, 2,
                                         kFaceReject, kRvaFaceRejectTgt,
                                         0xF8, &s_faceRejectCounter,
                                         &s_faceRejectPage);

      // [PortalWalk]: every portal the flood iterates, by target area. The
      // probe that separates "rejected by something I have not found" from
      // "never reached at all".
      //   1802eb93f  48 89 8D F8 00 00 00   mov [rbp+0F8h], rcx
      static constexpr std::uint8_t kPortalWalk[7] = {
        0x48, 0x89, 0x8D, 0xF8, 0x00, 0x00, 0x00,         // mov [rbp+0F8h], rcx
      };
      static_assert(sizeof(kPortalWalk) <= kIslandMaxSteal, "stub buffer");
      static constexpr std::uint8_t kMovRaxRcx[3] = { 0x48, 0x89, 0xC8 };
      any |= InstallRegBitIsland("PortalWalk", kRvaPortalWalk,
                                 sizeof(kPortalWalk), kPortalWalk,
                                 kMovRaxRcx,
                                 static_cast<std::int32_t>(kPortalBitsBytes * 8u - 1u),
                                 kPortalBitsOffset, kPortalBitsBytes,
                                 &s_portalWalkCounter, &s_portalWalkPage);

      // [SelWrite]: which areas actually get a selector written. Records r10
      // (the target area) rather than a portal index, so it uses the 32-byte
      // area table and a 0xFF mask.
      //   1802ec739  42 89 84 92 1C F9 1F 01   mov [rdx+r10*4+11FF91Ch], eax
      static constexpr std::uint8_t kSelWrite[8] = {
        0x42, 0x89, 0x84, 0x92, 0x1C, 0xF9, 0x1F, 0x01,
      };
      static_assert(sizeof(kSelWrite) <= kIslandMaxSteal, "stub buffer");
      static constexpr std::uint8_t kMovRaxR10[3] = { 0x4C, 0x89, 0xD0 };
      any |= InstallRegBitIsland("SelWrite", kRvaSelWrite,
                                 sizeof(kSelWrite), kSelWrite,
                                 kMovRaxR10, 0xFF,
                                 kSelAreaBitsOff, kSelAreaBitsSize,
                                 &s_selWriteCounter, &s_selWritePage);
      return any;
    }

    std::once_flag s_onceEd900Probe;
    std::atomic<bool> s_ed900ProbeInstalled{false};

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

  bool EnsureDispProbeHookInstalled() {
    std::call_once(s_onceDispProbe, []() {
      s_dispProbeInstalled.store(DoInstallDispProbe(), std::memory_order_release);
    });
    return s_dispProbeInstalled.load(std::memory_order_acquire);
  }

  bool EnsureEd900ProbeInstalled() {
    std::call_once(s_onceEd900Probe, []() {
      s_ed900ProbeInstalled.store(DoInstallEd900Probe(), std::memory_order_release);
    });
    return s_ed900ProbeInstalled.load(std::memory_order_acquire);
  }

  // Read-and-reset the island counter. Returns 0 when the hook is not
  // installed, which is indistinguishable from "ED900 never ran" — so the
  // caller must check the INSTALLED log line before reading a 0 as a result.
  // Same trap as [OccProbe] v1: a diagnostic that reports a constant because
  // nothing writes it is a probe defect, not a finding.
  std::uint64_t DrainEd900Count() {
    if (s_ed900Counter == nullptr)
      return 0;
    const std::uint64_t v = *s_ed900Counter;
    *s_ed900Counter = 0;
    return v;
  }

  std::uint64_t DrainEb620Count() {
    if (s_eb620Counter == nullptr)
      return 0;
    const std::uint64_t v = *s_eb620Counter;
    *s_eb620Counter = 0;
    return v;
  }

  std::uint64_t DrainEd900DropCount() {
    if (s_ed900DropCounter == nullptr)
      return 0;
    const std::uint64_t v = *s_ed900DropCounter;
    *s_ed900DropCounter = 0;
    return v;
  }

  bool Ed900DropProbeInstalled() {
    return s_ed900DropCounter != nullptr;
  }

  std::uint64_t DrainClipDegenACount() {
    if (s_clipDegenACounter == nullptr)
      return 0;
    const std::uint64_t v = *s_clipDegenACounter;
    *s_clipDegenACounter = 0;
    return v;
  }

  std::uint64_t DrainClipDegenBCount() {
    if (s_clipDegenBCounter == nullptr)
      return 0;
    const std::uint64_t v = *s_clipDegenBCounter;
    *s_clipDegenBCounter = 0;
    return v;
  }

  bool ClipDegenProbeInstalled() {
    return s_clipDegenACounter != nullptr && s_clipDegenBCounter != nullptr;
  }

  // Read-and-clear the portal bit-set, resolving each recorded bit to the
  // NEIGHBOUR AREA the abandoned crossing would have reached.
  //
  // The bit index is var_7B8 = portalIdx*3, and 0x2EC708-0x2EC716 reads the
  // neighbour as `word [qword_181748D00 + var_7B8*4 + 6]`, so the same
  // arithmetic applies directly to the bit index. Areas are returned rather
  // than portal indices because the area id is what [AreaDump] and [AreaSeed]
  // are already expressed in — 113/141/148/153 is the group to look for.
  // [QueueProbe] readers. Both follow the house rule: a zero from a probe that
  // failed to install is indistinguishable from a zero it measured, so each
  // exposes an Installed() predicate and the emitter prints it on the line.
  bool QueueProbeInstalled() {
    return s_qStartCounter != nullptr && s_qSkipCounter != nullptr;
  }

  // EAD60's return value, latched (not summed). sub_1802EB620 runs once per
  // VIEW, so this is the last view's cursor within the frame; read it with
  // eb620 on the same line, exactly as a20 is.
  std::uint32_t ReadQueueStartIndex() {
    if (s_qStartPage == nullptr)
      return 0xFFFFFFFFu;
    return *reinterpret_cast<volatile std::uint32_t*>(s_qStartPage + kQValueOffset);
  }

  std::uint64_t DrainQStartCount() {
    if (s_qStartCounter == nullptr)
      return 0;
    const std::uint64_t v = *s_qStartCounter;
    *s_qStartCounter = 0;
    return v;
  }

  std::uint64_t DrainQSkipCount() {
    if (s_qSkipCounter == nullptr)
      return 0;
    const std::uint64_t v = *s_qSkipCounter;
    *s_qSkipCounter = 0;
    return v;
  }

  // Areas that were in the order list, reached by the cursor, and skipped
  // because no portal crossing had written them a selector. Read and cleared.
  // This is the IDENTITY half: if 113/141/148/149/153 appear here, the cursor
  // is innocent and the loss is in the crossings; if they do not appear at all
  // the loop never reached them and ReadQueueStartIndex() is the answer.
  std::uint32_t DrainQSkipAreas(std::uint32_t* out, std::uint32_t maxOut) {
    if (s_qSkipPage == nullptr || out == nullptr || maxOut == 0)
      return 0;
    auto* bits = reinterpret_cast<volatile std::uint64_t*>(s_qSkipPage + kQAreaBitsOff);
    std::uint32_t n = 0;
    for (std::uint32_t w = 0; w < kQAreaBitsSize / sizeof(std::uint64_t); ++w) {
      std::uint64_t v = bits[w];
      bits[w] = 0;
      while (v != 0 && n < maxOut) {
        unsigned long b = 0;
        _BitScanForward64(&b, v);
        v &= v - 1;
        out[n++] = w * 64u + static_cast<std::uint32_t>(b);
      }
    }
    return n;
  }

  // Shared by [ClipDegen] and [FaceReject]: turn a recorder page's portal-bit
  // bitmap into neighbour AREA ids, read-and-clear. Factored out when the
  // second recorder landed so the two cannot drift apart in how they resolve
  // an area -- the identity is the whole point of both probes, and two copies
  // of this lookup is two chances to resolve it differently.
  // `maxBitOut` receives the widest bit index seen, which is the aliasing
  // guard: near the top of the map the ids on the line are fiction.
  // DEDUPED BY AREA, AND THE SENTINEL IS FILTERED. The first version emitted
  // one entry per portal BIT, which made the output useless for the question it
  // was built to answer:
  //   walkAreas=[62,63,63,64,65,180,180,180,180,180,180,61,63,74,126,180,...]
  // Distinct areas there are a handful, but the buffer is 48 and the scan runs
  // in bit order, so duplicates (two portals to the same neighbour) and ~30
  // copies of 180 filled it before the scan ever reached the high bits. Every
  // area behind that point read as ABSENT -- and "absent" is precisely the
  // verdict [PortalWalk] exists to deliver. A truncated identity list is worse
  // than none: it manufactures the answer it was asked for.
  //
  // 180 is not an area. dword_181748D8C is the area count (179), and the loop
  // at 0x2EC9D1 indexes nAreas+1, so ids run 0..nAreas with nAreas+1 as the
  // "no neighbour / outside" terminator. It is counted into `oobOut` rather
  // than listed, so the line still shows it exists without it drowning the ids
  // that matter. rtx.conf's warning about "area 180 on a map with 179 areas"
  // was about bit ALIASING; this is the benign source of the same number, and
  // maxBit still guards the aliasing case separately.
  //
  // Dedupe is by a 256-bit set, so the output is at most one entry per area and
  // the caller's buffer can no longer bind in practice.
  //
  // The area count comes from the EXISTING kRvaAreaCount in the anonymous
  // namespace above (0x1748D8C, "dword_181748D8C, its bound") -- the same
  // constant [AreaSeed] already bounds the order-list scan with. A second
  // definition here was ambiguous at this scope and, worse, would have been a
  // second place for the RVA to drift.
  static std::uint32_t DrainPortalAreasFromPage(std::uint8_t* page,
                                                std::uint32_t* out,
                                                std::uint32_t maxOut,
                                                std::uint32_t* maxBitOut,
                                                std::uint32_t* oobOut) {
    if (maxBitOut != nullptr)
      *maxBitOut = 0;
    if (oobOut != nullptr)
      *oobOut = 0;
    if (page == nullptr || out == nullptr || maxOut == 0)
      return 0;
    HMODULE client = GetModuleHandleA("client.dll");
    if (client == nullptr)
      return 0;

    auto* bits = reinterpret_cast<volatile std::uint64_t*>(
      page + kPortalBitsOffset);
    const auto words =
      static_cast<std::uint32_t>(kPortalBitsBytes / sizeof(std::uint64_t));

    const auto* pTable = reinterpret_cast<const std::uint8_t* const*>(
      reinterpret_cast<std::uint8_t*>(client) + kRvaPortalTable);
    const std::uint8_t* table =
      WvReadable(pTable, sizeof(void*)) ? *pTable : nullptr;

    // Area count, for the sentinel test. If it cannot be read, nothing is
    // filtered -- better a noisy list than one silently missing real ids.
    const auto* pAreaCount = reinterpret_cast<const std::uint32_t*>(
      reinterpret_cast<std::uint8_t*>(client) + kRvaAreaCount);
    const std::uint32_t areaCount =
      WvReadable(pAreaCount, sizeof(std::uint32_t)) ? *pAreaCount : 0xFFFFFFFFu;

    std::uint64_t seen[4] = { 0, 0, 0, 0 };         // 256 areas
    std::uint32_t n = 0;
    std::uint32_t oob = 0;
    std::uint32_t maxBit = 0;
    for (std::uint32_t w = 0; w < words; ++w) {
      std::uint64_t v = bits[w];
      if (v == 0)
        continue;
      bits[w] = 0;                                  // read and clear
      {
        unsigned long hi = 0;
        _BitScanReverse64(&hi, v);
        maxBit = w * 64u + static_cast<std::uint32_t>(hi);
      }
      // NOTE the bit walk is NOT bounded by maxOut: every bit must be visited
      // so `oob` and the dedupe stay honest even if the caller's buffer fills.
      while (v != 0) {
        unsigned long b = 0;
        _BitScanForward64(&b, v);
        v &= v - 1;
        const std::uint32_t bit = w * 64u + static_cast<std::uint32_t>(b);
        std::uint32_t area = 0xFFFFFFFFu;           // unresolved
        if (table != nullptr) {
          const void* slot = table + static_cast<std::size_t>(bit) * 4u + 6u;
          if (WvReadable(slot, sizeof(std::uint16_t)))
            area = *reinterpret_cast<const std::uint16_t*>(slot);
        }
        if (area >= areaCount || area >= 256u) {    // sentinel / unresolved
          ++oob;
          continue;
        }
        const std::uint64_t mask = 1ull << (area & 63u);
        if ((seen[area >> 6] & mask) != 0)
          continue;                                 // already listed
        seen[area >> 6] |= mask;
        if (n < maxOut)
          out[n++] = area;
      }
    }
    // Published even when the caller's buffer filled, because the aliasing
    // question is about the WIDEST index seen, not the first sixteen.
    if (maxBitOut != nullptr)
      *maxBitOut = maxBit;
    if (oobOut != nullptr)
      *oobOut = oob;
    return n;
  }

  std::uint32_t DrainClipDegenAreas(std::uint32_t* out, std::uint32_t maxOut) {
    std::uint32_t maxBit = 0;
    const std::uint32_t n =
      DrainPortalAreasFromPage(s_clipDegenAPage, out, maxOut, &maxBit, nullptr);
    dxvk::tf2::g_clipDegenMaxBit.store(maxBit, std::memory_order_relaxed);
    return n;
  }

  // [FaceReject] — portals skipped at 0x2EB98F because the camera is behind
  // the portal plane, by TARGET AREA. Read and cleared.
  //
  // READ IT AGAINST qSkipAreas ON THE SAME FRAME. Areas 92/113/141/148/153
  // appearing here on the LOW side and not on the HIGH side closes the chain:
  // camera crosses the plane -> this test flips -> the portal to those areas
  // is skipped -> no selector is written -> the queue loop skips them at
  // 0x2EB884 -> a20 falls 14 -> 9 -> m1 and m2 both halve.
  // Absent here while still in qSkipAreas would mean the crossing dies
  // somewhere else again, and the enumeration missed a seventh reject.
  bool FaceRejectProbeInstalled() { return s_faceRejectCounter != nullptr; }

  std::uint64_t DrainFaceRejectCount() {
    if (s_faceRejectCounter == nullptr)
      return 0;
    const std::uint64_t v = *s_faceRejectCounter;
    *s_faceRejectCounter = 0;
    return v;
  }

  std::uint32_t DrainFaceRejectAreas(std::uint32_t* out, std::uint32_t maxOut) {
    return DrainPortalAreasFromPage(s_faceRejectPage, out, maxOut,
                                    nullptr, nullptr);
  }

  // [PortalWalk] — the target area of EVERY portal the flood iterated.
  //
  // THE DISCRIMINATION, and it needs no threshold: for an area that ends up
  // selector-less (in qSkipAreas),
  //   present here  -> the portal WAS walked and something declined to enqueue
  //                    it, so a reject exists that six instrumented sites do
  //                    not cover and the enumeration is incomplete.
  //   absent here   -> nothing rejected it; its source area never ran its
  //                    portal loop. Confirmed cascade, and the question moves
  //                    to which source died first.
  bool PortalWalkProbeInstalled() { return s_portalWalkCounter != nullptr; }

  std::uint64_t DrainPortalWalkCount() {
    if (s_portalWalkCounter == nullptr)
      return 0;
    const std::uint64_t v = *s_portalWalkCounter;
    *s_portalWalkCounter = 0;
    return v;
  }

  // SOURCE -> TARGET pairs for every portal walked, deduped, read and cleared.
  //
  // WHY PAIRS. The target-only list settled that 92/113/141/148/153 are never
  // REACHED (walked 5 of 349 LOW frames against 126 of 134 HIGH) -- so no
  // reject is at fault and the loss is a cascade. It also showed eleven areas
  // (81,84,92,96,113,141,146,148,152,153,166) moving with identical counts to
  // the frame, which is one connected CLUSTER with a single entry, not eleven
  // independent failures. Targets alone cannot name that entry; the owning
  // area can.
  //
  // NO NEW INSTRUMENTATION. The source is recoverable from the bit already
  // recorded: portals are contiguous per area in qword_181748CF8 (count at +0,
  // start at +2), and the bit is portalIdx*3, so the owner is the one area
  // whose [start, start+count) contains bit/3. Adding a second captured
  // register to the island would have meant re-verifying flags, scratch and
  // steal at a site that is already correct and already validated in a
  // capture. A drain-time lookup over 179 areas costs nothing and cannot
  // perturb the thing being measured.
  //
  // Pairs are deduped, the no-neighbour sentinel is excluded from targets (it
  // is counted into oobOut), and a source that cannot be resolved is reported
  // as 0xFFFF rather than dropped, so a hole in the range table is visible
  // instead of silently thinning the list.
  // [SelWrite] — areas that actually received a selector at 0x2EC739, read and
  // cleared. Read against walkPairs and qSkipAreas on the same line:
  //   149 HERE in LOW  => it is enqueued and the cursor never returns for it;
  //                       the defect is ordering, and rewinds= says whether
  //                       the engine's own rewind even fired.
  //   149 ABSENT in LOW => the crossing is rejected by something upstream of
  //                       this store that no counter covers yet.
  // [DropAreas] — WHICH areas sub_1802ED900 dropped, read and cleared.
  //
  // NOTE there is deliberately no count drain here. s_dropAreaCounter is the
  // SAME counter DrainEd900DropCount() already reports as ed900Drop=; adding a
  // second reader would let whichever drained first zero it and the other read
  // 0. The count keeps its existing name on the line; this adds only identity.
  std::uint32_t DrainDropAreas(std::uint32_t* out, std::uint32_t maxOut) {
    if (s_dropAreaPage == nullptr || out == nullptr || maxOut == 0)
      return 0;
    auto* bits = reinterpret_cast<volatile std::uint64_t*>(
      s_dropAreaPage + kSelAreaBitsOff);
    std::uint32_t n = 0;
    for (std::uint32_t w = 0; w < kSelAreaBitsSize / sizeof(std::uint64_t); ++w) {
      std::uint64_t v = bits[w];
      bits[w] = 0;
      while (v != 0 && n < maxOut) {
        unsigned long b = 0;
        _BitScanForward64(&b, v);
        v &= v - 1;
        out[n++] = w * 64u + static_cast<std::uint32_t>(b);
      }
    }
    return n;
  }

  bool SelWriteProbeInstalled() { return s_selWriteCounter != nullptr; }

  std::uint64_t DrainSelWriteCount() {
    if (s_selWriteCounter == nullptr)
      return 0;
    const std::uint64_t v = *s_selWriteCounter;
    *s_selWriteCounter = 0;
    return v;
  }

  std::uint32_t DrainSelWriteAreas(std::uint32_t* out, std::uint32_t maxOut) {
    if (s_selWritePage == nullptr || out == nullptr || maxOut == 0)
      return 0;
    auto* bits = reinterpret_cast<volatile std::uint64_t*>(
      s_selWritePage + kSelAreaBitsOff);
    std::uint32_t n = 0;
    for (std::uint32_t w = 0; w < kSelAreaBitsSize / sizeof(std::uint64_t); ++w) {
      std::uint64_t v = bits[w];
      bits[w] = 0;
      while (v != 0 && n < maxOut) {
        unsigned long b = 0;
        _BitScanForward64(&b, v);
        v &= v - 1;
        out[n++] = w * 64u + static_cast<std::uint32_t>(b);
      }
    }
    return n;
  }

  // dword_1811FBD98, the engine's own cursor-REWIND counter (incremented at
  // 0x2EC8DC when an enqueued area's slot was behind the cursor). Free to read,
  // no hook. This is the number whose EXISTENCE was checked earlier and whose
  // VALUE never was -- which is why the ordering theory got dropped too soon.
  // Cumulative since process start; difference it across frames.
  std::uint32_t ReadRewindCount() {
    HMODULE client = GetModuleHandleA("client.dll");
    if (client == nullptr)
      return 0xFFFFFFFFu;
    const auto* p = reinterpret_cast<const std::uint32_t*>(
      reinterpret_cast<std::uint8_t*>(client) + kRvaRewindCount);
    return WvReadable(p, sizeof(std::uint32_t)) ? *p : 0xFFFFFFFFu;
  }

  std::uint32_t DrainPortalWalkPairs(std::uint32_t* srcOut,
                                     std::uint32_t* dstOut,
                                     std::uint32_t maxOut,
                                     std::uint32_t* oobOut) {
    if (oobOut != nullptr)
      *oobOut = 0;
    if (s_portalWalkPage == nullptr || srcOut == nullptr || dstOut == nullptr ||
        maxOut == 0)
      return 0;
    HMODULE client = GetModuleHandleA("client.dll");
    if (client == nullptr)
      return 0;
    auto* base = reinterpret_cast<std::uint8_t*>(client);

    const auto* pTable = reinterpret_cast<const std::uint8_t* const*>(
      base + kRvaPortalTable);
    const std::uint8_t* table =
      WvReadable(pTable, sizeof(void*)) ? *pTable : nullptr;
    const auto* pRange = reinterpret_cast<const std::uint8_t* const*>(
      base + kRvaAreaPortalRange);
    const std::uint8_t* range =
      WvReadable(pRange, sizeof(void*)) ? *pRange : nullptr;
    const auto* pAreaCount =
      reinterpret_cast<const std::uint32_t*>(base + kRvaAreaCount);
    const std::uint32_t areaCount =
      WvReadable(pAreaCount, sizeof(std::uint32_t)) ? *pAreaCount : 0u;

    // Reverse map portalIdx -> owning area, built once per drain from the same
    // (count, start) pairs the game itself indexes with.
    constexpr std::uint32_t kMaxAreas = 256;
    std::uint16_t start[kMaxAreas] = {};
    std::uint16_t count[kMaxAreas] = {};
    const std::uint32_t nAreas =
      (range != nullptr && areaCount != 0 && areaCount <= kMaxAreas) ? areaCount : 0;
    for (std::uint32_t a = 0; a < nAreas; ++a) {
      const void* rec = range + static_cast<std::size_t>(a) * 8u;
      if (!WvReadable(rec, 4)) {
        count[a] = 0;
        continue;
      }
      count[a] = *reinterpret_cast<const std::uint16_t*>(rec);
      start[a] = *(reinterpret_cast<const std::uint16_t*>(rec) + 1);
    }

    auto* bits = reinterpret_cast<volatile std::uint64_t*>(
      s_portalWalkPage + kPortalBitsOffset);
    const auto words =
      static_cast<std::uint32_t>(kPortalBitsBytes / sizeof(std::uint64_t));

    // Dedupe on the (source, target) pair. 256x256 bits = 8KB is far more than
    // is needed, so the pair set is kept as a small linear scan over what has
    // already been emitted -- maxOut is 64 and the real pair count is ~25.
    std::uint32_t n = 0;
    std::uint32_t oob = 0;
    for (std::uint32_t w = 0; w < words; ++w) {
      std::uint64_t v = bits[w];
      if (v == 0)
        continue;
      bits[w] = 0;                                   // read and clear
      while (v != 0) {
        unsigned long b = 0;
        _BitScanForward64(&b, v);
        v &= v - 1;
        const std::uint32_t bit = w * 64u + static_cast<std::uint32_t>(b);

        std::uint32_t dst = 0xFFFFFFFFu;
        if (table != nullptr) {
          const void* slot = table + static_cast<std::size_t>(bit) * 4u + 6u;
          if (WvReadable(slot, sizeof(std::uint16_t)))
            dst = *reinterpret_cast<const std::uint16_t*>(slot);
        }
        if (areaCount != 0 && (dst >= areaCount || dst >= 256u)) {
          ++oob;                                     // sentinel / unresolved
          continue;
        }

        const std::uint32_t portalIdx = bit / 3u;
        std::uint32_t src = 0xFFFFu;                 // unresolved, still listed
        for (std::uint32_t a = 0; a < nAreas; ++a) {
          if (count[a] != 0 && portalIdx >= start[a] &&
              portalIdx < static_cast<std::uint32_t>(start[a]) + count[a]) {
            src = a;
            break;
          }
        }

        bool dup = false;
        for (std::uint32_t i = 0; i < n; ++i) {
          if (srcOut[i] == src && dstOut[i] == dst) {
            dup = true;
            break;
          }
        }
        if (dup)
          continue;
        if (n < maxOut) {
          srcOut[n] = src;
          dstOut[n] = dst;
          ++n;
        }
      }
    }
    if (oobOut != nullptr)
      *oobOut = oob;
    return n;
  }

  // Read-and-clear the [DegenPair] (r11, r9) histogram from the ClipDegenA
  // recorder. `out` receives kDegenPairCells entries, cell = r9*4 + r11 with
  // r11 clamped at 3 and r9 at 15.
  //
  // WHAT TO READ. Every entry is one portal abandoned at 0x2EC675, labelled by
  // the two counts that would have become rec[+0] (edges, r11) and rec[+2]
  // (planes, r9). The question the capture has to settle is whether relaxing
  // ONLY `cmp r11,3` can do anything:
  //
  //   cells with r9 >= 3  -> those portals would survive the untouched
  //                          `cmp r9,3` at 0x2EC67B, reach the allocator at
  //                          0x2EC6FA, and enqueue their neighbour area. The
  //                          r11-only patch is worth applying, and it cannot
  //                          produce the rec[+2]==0 record that crashed
  //                          sub_1802ED900 — the surviving r9 gate is the
  //                          guarantee, not an assumption.
  //   cells with r9 < 3   -> the reject just moves four bytes down to
  //                          0x2EC67F. The r11-only patch is a no-op and the
  //                          intervention has to go upstream, into whatever
  //                          makes the clip degenerate in the first place.
  //
  // r11 == 3 must stay empty; see the self-check note on the stub.
  std::uint32_t DrainDegenPairs(std::uint64_t* out, std::uint32_t maxOut) {
    if (s_clipDegenAPage == nullptr || out == nullptr)
      return 0;
    const std::uint32_t n =
      maxOut < kDegenHistCells ? maxOut : static_cast<std::uint32_t>(kDegenHistCells);
    auto* cells = reinterpret_cast<volatile std::uint64_t*>(
      s_clipDegenAPage + kDegenHistOffset);
    for (std::uint32_t i = 0; i < n; ++i) {
      out[i]   = cells[i];
      cells[i] = 0;                                   // read and clear
    }
    return n;
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
