#include "tf2_decal_hook.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <windows.h>

#include "../util/log/log.h"
#include "../util/util_string.h"

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
