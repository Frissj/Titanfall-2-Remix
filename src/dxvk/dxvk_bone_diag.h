// NV-DXVK NPC SKINNING DIAG — shared declarations for the bone-pipeline
// diagnostic logging used while debugging Source-engine character bone
// uploads on dxvk-remix. Master switch is `tf2::boneDiagEnabled()`,
// which reads the `RTX_BONE_DIAG` env var once at first call.
//
// All logging hooks ([BoneSrvs], [BoneCacheSweep], [Dxvk.copyBufTo*],
// [Dxvk.updateBuf*], [BoneMap.def], [BoneTargetArm], [BoneTargetFrame],
// [BonePerFrameByBuf], [interleaver.skin*]) are gated on this; the
// build is silent by default.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace dxvk { namespace tf2 {
  // Master on/off switch — set RTX_BONE_DIAG=1 to enable.
  bool boneDiagEnabled();

  // Per-buffer per-frame stats, snapshotted + cleared by EndFrame.
  struct PerBufFrameStats {
    uint32_t copyWrites = 0;
    uint32_t updateWrites = 0;
    uint32_t copyBytes = 0;
    uint32_t updateBytes = 0;
    uint32_t copySharedRot = 0;
    uint32_t updateSharedRot = 0;
    uint32_t reads = 0;
  };
  extern std::mutex g_perBufStatsMutex;
  extern std::unordered_map<uintptr_t, PerBufFrameStats> g_perBufStats;

  // Per-frame counters for the targeted bone-matrix buffer (selected
  // via RTX_NPC_BONE_BUF env var, or auto-armed by [BoneSrvs] from the
  // first sameBuf=0 NPC draw). Reset and reported by EndFrame.
  extern std::atomic<uint32_t> g_targetCopyBufThisFrame;
  extern std::atomic<uint32_t> g_targetUpdateBufThisFrame;
  extern std::atomic<uint32_t> g_targetCopyBufBytesThisFrame;
  extern std::atomic<uint32_t> g_targetUpdateBufBytesThisFrame;
  extern std::atomic<uint32_t> g_targetReadDispatchesThisFrame;
  extern std::atomic<uint32_t> g_frameCounterForTarget;

  // Auto-armed by [BoneSrvs] on the first sameBuf=0 NPC draw it sees.
  extern std::atomic<uintptr_t> g_autoTargetBufPtr;

  // Per-frame chronological trace of writes & reads against the targeted
  // bone-matrix buffer. EndFrame dumps and clears. Lets us see, in one log,
  // whether a real-rig copyBuffer write is followed (and stomped) by a
  // filler updateBuffer write for the same palette range BEFORE the draw
  // that reads that palette.
  struct BoneOp {
    uint64_t seq;        // global monotonic sequence
    char     op;         // 'W' = write (copyBuf/updateBuf), 'R' = read (dispatch)
    uint32_t offset;     // byte offset within the 393216-byte t30 buffer
    uint32_t size;       // bytes of this op (48 per bone for writes; slice
                         // length for reads)
    uint32_t bones;      // size / 48 (writes), or slice/48 (reads)
    uint8_t  sharedRot;  // for writes, true if first and last bone share R0
                         // (filler pattern). Always 0 for reads.
    uint8_t  source;     // for writes: 1=copyBuf, 2=updateBuf. For reads: 3.
    char     tag[32];    // short VS hash for reads; empty for writes
  };
  extern std::mutex              g_boneTimelineMutex;
  extern std::vector<BoneOp>     g_boneTimeline;
  extern std::atomic<uint64_t>   g_boneTimelineSeq;

  // Per-frame record of SRV bindings seen by [BoneSrvs] for t30-sized
  // buffers. Lets EndFrame cross-reference: for each (buf, VS, FirstElem)
  // tuple observed during the frame, did the bone palette that tuple
  // samples actually receive a real-rig copyBuffer write this frame? If
  // not, the draw is reading filler — that's the T-pose signature. The
  // cross-check only resolves authoritatively for the *targeted* buffer
  // (because only that buffer has a full [BoneTimeline]); others get a
  // count-only summary for manual correlation.
  struct BoneSrvRecord {
    uintptr_t bufPtr;       // DxvkBuffer* for the t30 SRV's underlying buffer
    uint32_t  firstElem;    // SRV FirstElement (in bones)
    uint32_t  numElem;      // SRV NumElements (in bones)
    uint32_t  drawCount;    // how many draws this frame had this binding
    char      vsShort[24];  // short VS hash (first 19 chars + null)
    uint8_t   sameBuf;      // 1 if t30 == t32 (player viewmodel-style)
  };
  extern std::mutex                      g_boneSrvsMutex;
  extern std::vector<BoneSrvRecord>      g_boneSrvsThisFrame;

  // Capture + symbolicate the current call stack (Windows-only). Returns
  // a formatted multi-line string suitable for logging. Skips the top
  // `framesToSkip` frames (to hide the helper itself). Result is cached
  // by stack-fingerprint hash so repeat sites emit once; subsequent
  // calls with the same fingerprint return a short summary. Safe to
  // call from any thread — internally serializes around dbghelp.
  std::string captureBoneStackTrace(uint32_t framesToSkip = 1);

  // Returns true the FIRST time a given stack fingerprint is seen this
  // session, false afterwards. Use to gate heavy per-call-site logging.
  bool registerBoneStackSiteOnce(uint64_t stackFingerprint);
}}
