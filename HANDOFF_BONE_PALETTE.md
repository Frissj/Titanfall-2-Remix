# HANDOFF: bone-palette path — state, measurements, remaining work

Session date: 2026-07-23. Continues `HANDOFF_PERF_OPTIMIZATION.md`; everything here
is in the **working tree, uncommitted**. Every claim below has a log tag you can
re-verify. Where I predicted a number and missed, the miss is recorded.

---

## 1. The one fact that reframes this whole area

**The CPU bone palette does not skin anything in TF2.**

- The INTERLEAVER performs the bone-weighted skinning GPU-side, binding the
  game's own bone buffer as `INTERLEAVE_GEOMETRY_BINDING_BONE_MATRIX`
  (`rtx_geometry_utils.cpp` ~1932). It never reads `pBoneMatrices`.
- The only renderer consumer of `pBoneMatrices` is `dispatchSkinning` — the
  legacy fixed-function path — and `rtx_scene_manager.cpp` ~2805 gates it to
  `R32G32B32_SFLOAT` / `R32G32B32A32_SFLOAT` normals. TF2's normals are
  **fmt 98** (`R32_UINT`), so it never runs. Confirmed live:
  `[BonePaletteShare] ... legacySkin=0 normFmt=98`.

**But the palette is not dead** — it fed `computeHash()`, and `boneHash` is
load-bearing:
- `rtx_scene_manager.cpp:739` — `boneHash == lastBoneHash` decides
  `kUpdateInstance` (no re-skin) vs `kUpdateBVH` (re-skin). Get this wrong and
  animation freezes or re-skins every frame.
- `rtx_draw_call_cache.cpp:49/76/111` — BLAS matching and dedup.

So the palette was 256 `float3x4`→`Matrix4` conversions per skinned draw
existing solely to produce a change-detector.

---

## 2. What is implemented now

| Change | File | State |
|---|---|---|
| `BonePalette` COW wrapper replaces `std::vector<Matrix4>` in `SkinningData` | `rtx_types.h` | live |
| `boneHash` computed from **source bytes**, not the converted palette | `d3d11_rtx.cpp` | live |
| Palette materialised only when a consumer needs it (`needPalette`) | `d3d11_rtx.cpp` | live |
| `FirstElement` rebase | `d3d11_rtx.cpp` | **MEASURED, DISABLED** (`kApplyFirstElementRebase=false`) |
| Per-region cache generations + 8-entry share cache | `d3d11_rtx.{h,cpp}` | live but **DORMANT** (`builds=0`) |
| `read_bone_transform.h` bounds on palette size, not `numBones` | graph component | live |
| `.vec()` / `mutableVec()` call-site fixes | `rtx_game_capturer.cpp`, `rtx_remix_api.cpp` | live |

`needPalette` is true only for: float normals (legacy skinning would run),
`numBones == 1` (the single-bone bake at `rtx_types.cpp:533`), or an active USD
capture. On TF2 it is **always false**.

---

## 3. Measured results — including where I was wrong

| Bucket | before | after | note |
|---|---|---|---|
| `bonePalette` (was misnamed `cbc_rangeDiag`) | 16.5–23.2 µs/draw | **8.7–11.5** | ~2×. I predicted "near-zero" — **wrong** |
| `xt_cls` | 7.9 | **2.7–3.5** | staged cb reads + pointer-keyed log dedup |
| `fm_tail` | 5.7 | **4.1** | alpha-test RDEF memo + staged read |
| `indexSnap` | ~20 ms/frame | **~0** | replaced by in-stream GPU stash |

**Why `bonePalette` did not go to zero:** `builds=0` proves no palette is ever
converted, so the residual is *not* conversion. It is the **393 KB
mirror→`m_fullBoneCache` merge scan** sharing the same bucket — a byte-wise
non-zero test over the whole buffer, running whenever `g_boneCacheMirrorGen`
advances (~317 bumps per 1024 draws). That merge is now the target here.

**Bucket rename:** `cbc_rangeDiag` → `bonePalette`. The old name was actively
misleading — its `markStg` boundary sits far past the raw-UV diagnostic it was
named for, and that diagnostic is `RTX_D3D11_DIAG`-gated and free in a normal
run. Check what a bucket *spans* before optimising it.

---

## 4. Verified by mechanism (thermal-independent)

This machine swung ~4× on identical code during the session, so timer-based
conclusions were repeatedly worthless. These checks are not:

- `[BonePaletteShare] palOobReads=0` — **nothing** in the codebase indexes an
  unmaterialised palette. Every read funnels through `BonePalette::operator[]`,
  which counts out-of-range reads and returns identity.
- `[BonePaletteShare] builds=0` — palette never materialised on the TF2 path.
- `[SkinPalProbe]` silent — `dispatchSkinning` never runs undersized.
- `[ReskinProbe] draws=4424 boneHashChanged=2845 reskinned=2850 zeroBoneHash=0`
  — hashes change and meshes re-skin; nothing frozen by the hash-basis change.
- `[IdxStashPool] created=164` flat while `reused` climbs past 4000 — index
  stash pooling reaches steady state with zero allocations.

**Prefer this style of check.** `created`/`builds`/`palOobReads` settle a
question in one run regardless of thermal state; a µs bucket does not.

---

## 5. Remaining optimisations, ranked

1. **Mirror-merge scan** (now the dominant `bonePalette` cost). The merge walks
   all 393216 bytes in 48-byte chunks testing for non-zero, then copies. The
   producer (`g_boneCacheMirror`, written by `DxvkContext::copyBuffer`
   interception) knows the byte range it wrote — record dirty ranges there and
   merge only those, exactly as `BumpBoneCacheRegions` already does for
   invalidation. Both merge sites need it: `d3d11_rtx.cpp` ~23170 (per-draw) and
   ~35486 (end-of-frame sweep).
2. **Strip the dormant share cache.** The 8-entry ring, `windowGen`,
   `m_boneCacheRegionGen`, `BumpBoneCacheRegions` / `BoneCacheWindowGen` and the
   `mergeTouched`/`sweepTouched` tracking only run when `needPalette` is true —
   never, on TF2. It is ~80 lines across three files and a live footgun: any new
   writer of `m_fullBoneCache` must bump both the global and the region
   generations or stale palettes get served. I built it one step before
   discovering the palette shouldn't be built at all; deleting it is the honest
   cleanup. Keep a note so it isn't reinvented.
3. **`bones=256` on every draw.** The real palette size is not obtainable from
   the SRV (see §6). The only remaining route is `max(BLENDINDICES) + 1`, which
   needs a vertex scan — cacheable per immutable VB keyed on
   `(buffer ptr, offset, vertexCount)`, the same rule the objAabb cache uses.
   Risk: an unsound key truncates the palette and breaks skinning.

---

## 6. Do NOT re-attempt (refuted by measurement)

- **SRV `NumElements` as the model's bone count.** Tried; TF2 binds the whole
  buffer. `[BonePalette] srvNumElems=8192 srvBoneLimit=8192 bonesConverted=256`
  — zero reduction. The view extent says nothing about palette size.
- **Re-applying the `FirstElement` rebase without further work.** Honouring
  `FirstElement` is correct by D3D11 semantics, and it is real:
  `[BonePalette]` measures `srvFirstElem=80` and `=288` on live TF2 skinned
  VSes. But the palette doesn't feed rendering, so rebasing it fixes nothing
  observable — while, once `boneHash` was computed from that pointer, it *moved
  the hash window* and changed re-skin decisions. **Prerequisite:** the
  interleaver indexes `palette[BLENDINDICES + boneIndexBase]`
  (`rtx_geometry_utils.cpp` ~1803), so `FirstElement` alone does not determine
  where a draw's bones live. Establish the real window first.
- **Deleting `dispatchSkinning`.** Dead for TF2, live for other titles/formats,
  and free at runtime (format-gated).

---

## 7. Invariants

- `boneHash` may change **basis** freely (nothing compares it against a value
  produced outside this path) but must remain a faithful **change detector** for
  the bones a draw actually uses.
- `numBones` must stay non-zero for skinned draws — the accel manager uses it to
  pick the DYNAMIC BLAS path that handles per-frame refit.
- Palette readers must bound on `pBoneMatrices.size()`, never on `numBones`.
- `BonePalette` element access is **const-only** by design. A non-const
  `operator[]` would detach (COW) and deep-copy on every ordinary read,
  including per-draw diagnostics holding a non-const `DrawCallState`. Mutate
  only via `mutableVec()`.
- Any new writer of `m_fullBoneCache` must bump `m_fullBoneCacheGen` **and** the
  region generations (until §5.2 removes them).

---

## 8. Open

- **Broken-triangle artifact (screenshot, cave interior, shard/sliver mesh).**
  NOT attributed to the bone changes: all three probes clean (§4), and the
  centre-screen pick names `VS=0x2af9b90d63850ec3` =
  `VS_7c38fdf4358d5527…` — a **BSP world-geometry** shader (it is in the
  `vsIsBspAllowed` allowlist), 1.3 M pixels, absent from `[BonePalette]` i.e.
  never in the bone path. **Caveat:** the pick rect is 75×43 at screen centre and
  the shards surround that point, so it may be sampling the wall behind them.
  To close: aim `PickRegion2` directly at a shard, or enable `[SurfTrack]`
  (currently off — it emits nothing) for per-VS screen bboxes.
- The whole frame is pink/red tinted in that capture. That matches the
  "red corruption / garbage-o2w" family in the older notes, not skinning. If the
  tint is also new it is the stronger lead.
- `[ReskinProbe]` prints `vs=0x0` — I keyed it on
  `hashes[HashComponents::VertexShader]`, which is 0 for these draws. Aggregates
  are valid; per-VS attribution needs `transformData.vertexShaderHash`.
- `FirstElement=80` / `=288` remain measured but unexplained.

---

## 9. Probes added this session

| Tag | Where | Answers |
|---|---|---|
| `[BonePalette]` | `d3d11_rtx.cpp` | per-VS SRV geometry: `srvFirstElem`, `srvNumElems`, `bonesConverted`, `rebased` |
| `[BonePaletteShare]` | `d3d11_rtx.cpp` | `served/builds/needPalette/legacySkin/normFmt/palOobReads` |
| `[SkinPalProbe]` | `rtx_geometry_utils.cpp` | fires (error) if legacy skinning runs with a short palette |
| `[ReskinProbe]` | `rtx_scene_manager.cpp` | per-window `draws/boneHashChanged/reskinned/zeroBoneHash` |
| `[IdxStashPool]` | `rtx_scene_manager.cpp` | index-stash pool health: `created/reused/agedOut/freeBytes` |
