# How to Identify the Ridden Dropship (and the decoys that wasted days)

Companion to `CULLING_BIBLE.md`. The single biggest time-sink in this investigation was **misidentifying
which "ship" the logs were showing.** Three different objects all look like "the dropship," they live in
three different ID systems and coordinate frames, and chasing the wrong one produced confident-but-wrong
conclusions (the whole "studio LOD is the cull" theory was about a decoy). This note records the method
that finally pinned the *real* ridden ship, so we never re-walk that path.

---

## TL;DR — the rule that works (UPDATED 2026-06-08, SESSION-L/M)

**Identify the ridden ship by which widow renderable's world-AABB CONTAINS the camera** — the
camera-inside-AABB test. NOT name, position, handle, studio `dist²`, and **NOT "which renderable drops"**
(that's the FORMATION ship — see below). Two `hw=541` widow renderables exist: the **ridden** ship (camera
geometrically inside its AABB; studio origin `x≈0,z≈10000`; moves WITH the camera at a fixed offset) and a
**formation** dropship (camera OUTSIDE its AABB; flies a real path `x:300→1400`, ~1300u out).

> ⚠️ **`[DropTrace] present` is AMBIGUOUS — do NOT anchor on it.** It counts ALL Crow/widow instances
> (ridden + formation + phantom) lumped together, so `present 2→1` was the **formation** ship being
> *legitimately* frustum-culled, not the ridden ship vanishing. SESSION-G/H/I chased that and tracked the
> wrong ship for days. The CLEAN ridden-ship signal is **`[RiddenTrace] riddenSubmeshes`** (widow/Crow
> RtInstances at Remix `o2w.t` < 200u from the recentered origin): it goes `6→0` when the ridden ship
> actually vanishes. The ridden vanish is `riddenSubmeshes=0` (no RtInstance created), NOT a render-list drop.

`GetModelName` **crashes** on the real models and `GetModelIndex` returns −1, so **name-by-engine-call is
impossible.** Use the DXVK-side material name (`studioModelName`, captured at the studio draw) instead.

---

## The FOUR decoys (each one led us astray)

| # | Decoy | Looks like the ship because… | How it betrays itself |
|---|---|---|---|
| 1 | **The distant FLEET** — other widows flying in formation (`hw=540`, `hw=318`) | Same model, same VS, same studio shader; goes to LOD3 (near-empty 2-mesh) and "vanishes" | Its `dist²` is genuinely huge (Δy≈16000 from camera); its LOD is a *correct* distance-LOD. It vanishes by distance, **independent of your view-direction bug.** Its model origins move on smooth flight paths. |
| 2 | **The PHANTOM** — `widow_int01` at Source `(0,−15000,10097)` (= studio `hw=541`) | Sits right where the player is; "near the camera"; named widow | `hidden=1 mask=0x0 blasPrim=0` in **both** visible and despawned states — permanently masked. It never changes at the despawn, so it is **not** what the user sees. |
| 3 | **Pilot / viewmodel / cockpit** — `hw=510/511/434`, dist²≈2–3600 (≈2–60 units) | Closest things to the camera, always drawn at `v10=0` | Present every frame through the despawn, constant. They're the player body / weapon, not the ship. |
| 4 | **The FORMATION dropship** — a *second* `hw=541` widow, ~1300u out, flying (`x:300→1400`) **(SESSION-L)** | Same model/`hw` as the ridden ship; **it IS the one that drops from the render list** (`dropshipRenderables 2→1`, `[V9Probe] pre=1→mid=0`) | Camera is **outside** its AABB; `sub_1801A9C70` frustum-culls it *correctly* (its top-plane goes <0 as it flies off-screen). This is the decoy SESSION-G/H/I tracked for days. NOT your bug. |

Every one of these is "a dropship-ish thing near the camera or sharing the model" — which is exactly why
picking by name / proximity / handle / "it's a widow" / "it dropped from the list" went wrong for days.

---

## The real ridden ship — and the discriminators that found it

**The ridden ship = the `Crow_dropship` + `widow ext01/int01` RtInstances at Remix `o2w.t ≈ (0,0,0)`.**

Discriminators, in order of decisiveness:

0. **Camera-inside-AABB (SESSION-L — the one that actually works).** The ridden ship is the widow
   renderable whose **world-AABB contains the camera** (compare `[V9Probe] aabb=` to `[LodNear] view=` /
   the camera origin). It's the one you stand inside, so the camera is geometrically within its bounds and
   moves with it. The formation decoy (#4) has the camera OUTSIDE its AABB. This cleanly separates ridden
   from formation even though they share `hw=541` and both are "near-ish."

1. **`[RiddenTrace] riddenSubmeshes` (the clean vanish anchor).** Widow/Crow RtInstances at Remix `o2w.t`
   < 200u from the recentered origin; it goes `→0` exactly when the ridden ship vanishes. Use THIS, not
   `[DropTrace] present` (which lumps ridden+formation+phantom and tracks the formation decoy's frustum-cull).

2. **`o2w.t ≈ (0,0,0)` in REMIX world coords.** Remix recenters the world near the camera, so the ship the
   player is *inside* lands at the Remix origin — **not** at the Source-world altitude (`Y≈−15000`). The
   `(0,−15000,…)` instance is the phantom (decoy #2). Filter `[HullCensus]` by `o2w.t≈origin`, not by the
   Source position.

3. **Material name via the DXVK studio hook** (`studioModelName` = `Crow_dropship*` / `veh_air_widow_*`).
   This is the only reliable name source (engine getters crash).

4. **Geometry-count cross-check.** `[DropGeo]` index `count` matched `[StudioDump]` exactly
   (`ext01 count=80988` = 26996 faces). Matching the submitted index count to the dumped mesh confirms a
   draw really is the named submesh — not a same-name LOD/phantom.

---

## The trap that caused the multi-day detour: four ID systems that DON'T cross-map

A single physical ship shows up under four unrelated identifiers, in different coordinate frames. Treating
any one as "the ship's identity" is the mistake:

| ID system | Where it appears | Coordinate frame | Pitfall |
|---|---|---|---|
| **studiohwdata handle** `hw=` | `[LodAll]`/`[LodV10]`/`[LodNear]` (studio LOD selector) | Source studio coords (origins at `−15000`, `10000`, …) | One `hw` = one `.mdl`, **shared by every instance** of that model (the whole fleet + phantom + ridden share `hw=540/541`). A handle is a *model*, not a *ship*. |
| **`rend` instance ptr** | `[LodV10]` (after the SESSION-D add) | Source studio coords | Separates instances of one `hw`, but pointers recycle and don't survive long gaps. |
| **material name** `studioModelName` | `[DropTrace]`/`[DropGeo]`/`[HullCensus]` | n/a (string) | The only stable *name*, but lumps Crow+widow and all submeshes together. |
| **Remix `o2w.t` / world AABB** | `[HullCensus]` | **Remix** recentered world coords (`(0,0,0)`-ish) | Different frame from the studio coords above — `(0,0,0)` here ≠ `(0,−15000,…)` there. Conflating them is what made the phantom look like the ship and the ridden ship look "absent." |

**`dist²` in studio-LOD coordinates is a specific trap:** the ship's *studio render origin* can be far from
the *view origin* the LOD uses even though the player is sitting inside it — so the ridden ship can masquerade
as a "distant fleet member" at LOD3. Never infer "near/far/which ship" from studio `dist²` alone.

---

## Checklist for next time (don't repeat the detour)

- [ ] Identify the ridden ship by **camera-inside-AABB** (`[V9Probe] aabb=` vs camera origin) and the
      **`[RiddenTrace] riddenSubmeshes`** anchor. Do NOT anchor on `[DropTrace] present` — it lumps
      ridden+formation+phantom and tracks the formation decoy's *legitimate* frustum-cull (the SESSION-G/H trap).
- [ ] "Which renderable drops from the render list" = the FORMATION decoy (#4), NOT the ridden ship. The
      ridden ship is never render-list-culled; its vanish is `riddenSubmeshes=0` (no RtInstance), downstream
      in the studiorender draw worker (see CULLING_BIBLE §0 SESSION-M).
- [ ] Identify instances by `studioModelName` (DXVK hook), never by `GetModelName` (crashes).
- [ ] For the *ridden* ship, filter `[HullCensus]` by Remix `o2w.t ≈ (0,0,0)`; the `(0,−15000,…)` widow is
      the phantom.
- [ ] Confirm a draw is the named mesh by matching `[DropGeo] count` to `[StudioDump] idxCount`.
- [ ] Never trust one ID system. A `hw` handle is a model (shared); a name lumps submeshes; a Remix `o2w`
      is a different coordinate frame than studio coords. Cross-correlate at least two, gated by the
      despawn timing.
- [ ] Treat "it's a widow / it's near the camera / its LOD went to 3 / it's at the ship's altitude" as
      **insufficient** — those are exactly the three decoys' traits.
