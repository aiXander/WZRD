# Gig build plan — live region video + agent content tools (≈ 21 Nov 2026)

> **Status: in progress** — A1 done, A2 built (end-to-end gate open), E1 done (a–d), P1 + P4 + P3 done. Settled with Xander in a
> grilling session on 2026-10-10 (replacing the earlier "VJ agent — missing
> tools" proposal; git history has it), extended the same day (D12–D14), then
> cut into hand-off **work packages** with the cross-lane **contracts** pinned
> (review pass, D15–D16). Engine video design: [render-engine-roadmap.md](render-engine-roadmap.md)
> §5.9 — this plan sequences it. Current engine state:
> [../reference/render-engine.md](../reference/render-engine.md); UI rules:
> [../reference/operator-ui.md](../reference/operator-ui.md); MCP conventions:
> [../reference/mcp-server.md](../reference/mcp-server.md). Everything post-gig
> (effects library UI, effect chains, feedback/post, TAKE transitions, in-app
> generation, a setup room) is in [vj-toolbox.md](vj-toolbox.md) — don't pull
> it forward unless this plan is ahead of schedule.

## The target

A **lie-down concert**: a friend plays live guitar, Xander does live visuals.
Indoor room, **one projector**, flat walls + ceiling, plus **custom regions
such as hanging fabric**. Show model: a **pre-rendered TextureFlow library**
(rendered from venue photos, which Xander gets ≈ 2026-10-24) **topped up live**
by the agent during the show. Everything below serves that night; the long-term
goal (an opinionated autonomous VJ agent with eyes) shapes the choices but is
not built here.

Flat walls/fabric don't use the facade story pipeline (keyframes over a
building photo → background subtraction), so the drift-repair tools are
parked. The real gap is that **the engine can't play video**, so nothing
offline reaches the projector.

## Decisions (do not re-litigate)

| # | Decision | Why |
|---|---|---|
| D1 | **Native video in the engine** (roadmap §5.9), not Resolume via Syphon | One app on a base M2; region-first stays intact; agent render → region without leaving WZRD |
| D2 | **HAP only** for now; no H.264 path, **no compute-shader DXT fallback**. Request `TEXTURE_COMPRESSION_BC` **only if the adapter offers it**; without it the engine refuses video with a clear error, everything else keeps working | VJ standard (all-intra → perfect loops, instant seek, low CPU). Apple Silicon + Linux/NVIDIA both have BC. The one documented exception to the `Features::empty()` portability rule (CLAUDE.md + reference §1b) when it lands |
| D3 | Video is a **texture input that WGSL effects can read** (the §5.9 `videoClip` effect with an `image` input), not a standalone player layer | Shaders modulate video live (audio-reactive to the guitar). Same slot later serves stills / pack `references/` |
| D4 | **Region-space placement in the engine**: a clip bound to region X is scaled into X's **padded `source_box`** (the box `extract_color_regions` grows to a Nano-Banana-native aspect). Clips stay at rendered resolution and are scaled on the fly. Canvas-space (full-frame clip cut by the mask) is the other per-binding mode | Avoids N full-canvas, 90 %-black decodes; one clip reusable across regions. `reproject_video` stays for exporting full-canvas mp4s only |
| D5 | **`source_box` lives on the region** (in `pack.json`, contract C4). The clip's provenance sidecar records which region/box it was made for (shown in the clip grid, used to suggest a target). **No custom video-container metadata** | Placement belongs to the binding's target, not the clip |
| D6 | **MP4 → HAP transcode runs locally** in the MCP (ffmpeg here has `hap` + `dxv` encoders), codec a parameter | HAP is 5–10× bigger than H.264: only small files cross venue wifi |
| D7 | **Locked per-project canvas** chosen once from Nano Banana's native sizes (e.g. 1376×768); new projects only. Surface photos conformed at import; the pack is authored at canvas resolution; the engine scales canvas → projector and the 4-corner alignment absorbs the 1.792 vs 1.778 gap | "No drift by construction" (measurement below) |
| D8 | **Fix the aspect stretch** in `wzrd/utils.py`: stop treating < 2 % aspect mismatch as a match and stretching (`ASPECT_RATIO_TOLERANCE`); undo Nano Banana's crop exactly instead | Measured, below. Directly shifts fabric masks today |
| D9 | Regions come from **prompted Nano Banana segmentation maps** → `extract_color_regions` (Xander tested: works). Drawing regions on the projector output is a later add-on only if needed | — |
| D10 | Fresh agent clips land in the **clip grid** flagged NEW; the agent binds a clip only on **design**, and only when asked. **Taking live stays a human act** (existing rule). Agent is semi-autonomous for library building, with a **soft per-project spend cap** (default $20, in `project.json`; warn at 80 %, ask before exceeding) | First live show; autonomy later |
| D11 | Chat stays an **external Claude session** over MCP for the gig; in-app chat is later | — |
| D12 | The clip bin is a **region × clip-slot grid** (Resolume's layer rows, but rows are named regions). Each row is auto-filled from provenance sidecars (the region a clip was made for); clips without a region sit in an "unassigned" row; any clip can be dragged onto any row. Click a slot = bind that clip on that region in **design**; TAKE as usual. Compact copy of the grid in Perform | "Pick from the rendered assets for this region" is the core operator move |
| D13 | **In-app project switcher**: recent projects in the TopBar. The engine's scene dir (watcher, `effects/`, pack, `session.json`, `alignment.json`) is fixed at boot and `scene.load` only takes JSON, so switching = **re-root by restarting the in-process engine on the new scene path** (relaunch-with-restore measured ~160–290 ms, reference §1b), not a new RPC. Same mechanism as roadmap §5.11's "restart engine" button | Today a venue switch means relaunching from a terminal with `WZRD_SCENE` |
| D14 | **Seed ~10 clip effects** as soon as the video input works — kaleido, mirror, RGB split, pixelate, posterize, edge glow, noise displace, strobe gate, hue shift, luma key, each with declared dials. Authored into the **global effects library** (C6) and **copied** into the gig project's `effects/` (projects stay self-contained). Library *browser/tools* are post-gig ([vj-toolbox.md](vj-toolbox.md) phase 1) | A clip alone isn't playable live; these make the clips respond to the guitar |
| D15 | **Post-swap probation window** (roadmap §5.11) comes **into the gig** as package E2 — it was post-gig | The agent writes shaders live on show night; nothing else catches a shader that compiles but tanks the frame rate on live |
| D16 | **Global libraries live in this repo**: `library/effects/`, `library/recipes/`, `library/midi/`, git-versioned. `WZRD_EFFECTS_ROOT` / `WZRD_RECIPES_ROOT` / `WZRD_MIDI_ROOT` override each | A Finder-launched app never sees shell env vars, so the default is what actually runs. Keep entries small: large media (preview clips) gitignored, a `preview.jpg` committed |

**Drift measurement (2026-10-10, 48 keyframes + 11 clips from `test_imgs/stories/`
and `projects/`):** Nano Banana returns 1376×768 (1.792) for 16:9 surfaces —
fit-to-width, ≈ 3 px cropped top/bottom → a **constant 0.78 % vertical
stretch**, identical across samples/models, passed through by the 2 % tolerance.
Corrected, keyframes align to < 0.5 px; the proposal's "random 1–2 % zoom"
does not occur. Clips keyframe→keyframe don't drift (≤ 1 px); the one real
drift (0.44–0.73 %, a slow zoom) is a clip from the raw surface to a stretched
keyframe. On the kasbah facade the stretch lit 4.8 % of the frame as outline
specks after subtraction; corrected, 0.07 %. 7 of 48 keyframes were fully
re-framed by the model (gross failure, not drift). → `check_frames` gates
shift at ≈ 0.1 % (1–2 px), not 0.3 %.

## Shared contracts (pinned so the lanes can run in parallel)

Every package below builds against these, not against another lane's code.
A package that has to change one edits this section in the same commit and
says so in its summary; on landing, the contract moves into the reference
doc of its owner.

**C1 — `project.json`** — landed with P1: `{ "schema": 1, "canvas": [w, h],
"spend_cap_usd": 20 }`, absent = legacy project. Contract now in
[mcp-server.md](../reference/mcp-server.md) "The project folder and its
locked canvas".

**C2 — provenance sidecar** — landed with P3: `<output>.meta.json` beside
every media file a tool writes (`tool`, `args`, `inputs`, `parent`,
`created`, `est_usd`, `region`, `source_box`; `rating` reserved). Contract now
in [mcp-server.md](../reference/mcp-server.md) "Provenance sidecars and the
spend cap".

**C3 — clip files** (owner P5): a playable clip is a HAP `.mov` under the
project's `generated/`, stem shared with its source: `<stem>.mp4` (source),
`<stem>.mov` (HAP), `<stem>.mov.meta.json` (its own sidecar, `parent` = the
mp4, `region`/`source_box` copied, `est_usd` 0), `<stem>.thumb.jpg` (320 px
wide, middle frame). **The MCP writes the thumbnail at transcode time** — the
app has no MCP client before the toolbox, so it never generates one; a clip
without a thumb shows a placeholder.

**C4 — `source_box` in `pack.json`** — landed with P4: optional per-layer
`[x0, y0, x1, y1]`, pack pixels, max-exclusive like `bbox`, may extend past
the canvas. Contract now in [render-engine.md](../reference/render-engine.md) §2.2.

**C5 — clip binding in `scene.json`** — landed with E1: an `image`-type
effect input takes `{clip, placement: "region"|"canvas", rate}`, clip path
relative to the scene dir; placement is contain-fit, never stretched; the
playhead is stateless `(leg time × rate) mod duration`; a clip that can't
open is a per-binding warning, not a load error. Contract now in
[render-engine.md](../reference/render-engine.md) §2.1 (scene shape) and
§2.4 (`sample_image`, decode).

**C6 — global library layout** (D16): `library/effects/<name>/{shader.wgsl,
descriptor.json, preview.jpg}`, `library/recipes/<name>/{recipe.json,
style images, preview.jpg}`, `library/midi/<name>.midi.json`. Projects get
copies, never links.

## Work packages

**Lanes** = which part of the tree an agent owns. Packages in different
lanes run in parallel; packages in one lane run in order. Collision hot
spots when lanes overlap: `wzrd_mcp/tools_config.json` (every new tool adds
an entry — append, don't reorder), `wzrd_mcp/tools.py` (P4 and P5 touch
different functions), `wzrd-app/src-tauri/src/{lib,rpc}.rs` (A1, A2, MIDI —
serial).

| Week | PY lane (`wzrd/`, `wzrd_mcp/`) | ENG lane (`render-core/`) | APP lane (`wzrd-app/`) | FX + skills |
|---|---|---|---|---|
| 1 (from 10-13) | P1 → P3; P4 and P6 alongside | E1 step a | A1 | — |
| 2 | P2, P5 | E1 step b | A2 against C3/C5 with fixture files | — |
| 3 | P7 | E1 steps c–d | A2 finished on the real engine | F1 (from the day E1b plays a clip); library rendering starts (photos ≈ 10-24) |
| 4 | — | E2 | MIDI (optional) | F1 finished; S1 |
| 5 | Dress rehearsal: real fabric (venue or hung at home), full one-hour set without touching code | | | |
| 6 | Buffer, no new features | | | |

Each package: **lane · needs · files · build · gate**. "Needs" lists hard
dependencies only; a contract (C1–C6) is never a reason to wait.

### P1 — Locked canvas + aspect fix (D7, D8) — done 2026-10-10

`set_project(canvas=…)` writes C1; surface/region/pack tools author at the
canvas; every model-output → reference mapping undoes the model's crop
(`wzrd.utils.uncrop_*`) instead of stretching; region boxes pad to Nano
Banana native aspects; `nano_banana` defaults to `"auto"` with refs.
Gate: kasbah keyframes 4.1 px → 0.12–0.19 px residual; a synthetic fabric
seg map's masks land on ground truth (old stretch: 4 px at hems). Done — see
[mcp-server.md](../reference/mcp-server.md) "The project folder and its
locked canvas".

### P2 — `import_photos` + EXIF orientation — done 2026-10-10

`import_photos` (newest from `~/Downloads` or given paths → `captures/`,
upright, conformed to the canvas unless `conform=False`) and an upright
shared loader in `resolve_input` (HEIC via `pillow-heif`, EXIF rotation)
for every tool. Gate (synthetic, no real iPhone file on this machine): a
portrait HEIC and a sideways EXIF-6 JPEG both land upright at 1376×768.
Done — see [mcp-server.md](../reference/mcp-server.md) "The project folder
and its locked canvas".

### P3 — Provenance sidecars + spend cap — done 2026-10-10

`logged_tool` writes a C2 sidecar for every media output; `nano_banana`,
`kling_*` and `texture_flow` take `region=<layer id>` (recorded with its box)
and `allow_over_cap`, and refuse a call that would cross the project's spend
cap. Gate: a TextureFlow render with `region="region_1"` wrote a sidecar with
region + box `[-30, 2, 382, 734]` and $0.0122; over the cap → refused with
"$0.01 of its $0.01 cap". Done — see [mcp-server.md](../reference/mcp-server.md)
"Provenance sidecars and the spend cap". P5 builds on it: `to_hap` must
copy `region`/`source_box` into the `.mov` sidecar with `parent` = the mp4.

### P4 — `source_box` in the pack + starter `scene.json` — done 2026-10-10

`build_layerpack` carries each region's padded box from `islands.json` into
`pack.json` (C4) and writes a starter `scene.json` when the project has
none. Fixed on the way: `extract_color_regions`' `region_mask` pointed at
the box crop (packs stretched it over the canvas); it is now the full mask,
the crop is `region_crop_mask`. Gate: fresh canvas project → regions → pack;
all four boxes equal `islands.json`; the HEAD engine boots it (`4 layers @
1376x768`, 0 passes). Done — see [mcp-server.md](../reference/mcp-server.md)
"The project folder and its locked canvas".

### P5 — Clip pipeline: `to_hap`, thumbnails, `extract_frames`

PY · needs P3 (sidecar inheritance) · `wzrd_mcp/media_tools.py`,
`texture_flow` / `kling_*` download path. `to_hap(video, codec="hap"|"hap_q")`
via local ffmpeg → C3 files; pad dimensions to multiples of 4 (BC works in
4×4 blocks; ffmpeg's `hap` encoder refuses otherwise), keep fps. TextureFlow
and Kling run it automatically after download (`hap=True` default).
`extract_frames(video, at=[…|"first"|"last"], count=None)` → stills.
**Gate:** one TextureFlow render arrives as mp4 + HAP `.mov` + both
sidecars + thumb; the `.mov` loops in `ffplay` without a seam jump.

### P6 — `check_frames` — done 2026-10-10

`wzrd/check_frames.py` + the `check_frames` MCP tool: flash rate (> 3 large
flashes/s fails), loop seam, dark-floor haze, lit share, and with
`surface=` the residual shift (fails above 0.1 % of the long side). Gate:
synthetic strobes 1/3 Hz pass, 4/5 Hz fail; four real TextureFlow loops
pass (seam 0.73–1.27× a frame step); a kasbah keyframe 0.19 px passes, the
same shifted 3 px measures 4.35 px and fails. F1's per-effect flash gate
runs it on a recorded clip of each effect. Done — see
[mcp-server.md](../reference/mcp-server.md) file map.

### E1 — Engine video: HAP clips as an effect `image` input — DONE (2026-10-10)

Shipped in four commits: (a) the `image` input + `sample_image(uv)` +
built-in `videoClip`; (b) HAP decode — pure-Rust MOV reader + `snap`, one
decode thread + 4-slot staging ring per (plan, binding) stream,
`TEXTURE_COMPRESSION_BC` only if offered; (c) region placement from
`source_box` (`bbox` fallback), contain-fit; (d) MCP wiring
(`get_scene_context` lists clips with their sidecar region, warnings in
apply replies, C5 in `upsert_binding`) + the CLAUDE.md BC exception. All
gates passed on the M2 (seven clips and a seven-clip TAKE at 60 fps, p95
≤ 17.2 ms; six fabric + ceiling clips in their regions 60 fps; a cold
Claude session bound "the clip rendered for the third fabric" by its
sidecar region). What E2 inherits: a TAKE now opens no streams on live
(they move with the plan), but design's post-promote rebuild reopens them,
and a clip's first decoded frame lands one frame after its leg wakes. Done
— see [render-engine.md](../reference/render-engine.md) §1b / §2.1 / §2.4.

### E2 — Post-swap probation window (D15, roadmap §5.11)

ENG · needs E1 (same files, serial) · after any pipeline/plan swap on
**live**, watch ~30 frames; if p95 blows the budget and the swap is the
delta, auto-revert to the retained previous pipeline and emit `hot_reload`
`{reverted: "frame budget"}`. Trap: a TAKE crossfade (both legs rendering,
clip streams opening) legitimately raises frame time — attribute or exclude
it, or every big TAKE reverts itself. **Gate:** a deliberately slow shader
promoted to live reverts within ~0.5 s; a one-hour set of normal TAKEs
triggers zero reverts.

### A1 — Project switcher + "restart engine" (D13) — done 2026-10-10

Shipped: TopBar project menu + "Restart engine" (re-roots the in-process
engine on the same windows, ~150–250 ms; per-machine `settings.json`
recents; boot falls back to the most recent project). Gate passed in the
browser seat: fixture-a ⇄ fixture-b each came back with its own masters,
alignment and looks; a `WZRD_SPIKE=panic` crash → one click restored.
Needed a small render-core fix (OSC/WS listeners release their ports on
drop, `osc.rs`/`ws.rs`). Done — see
[../reference/operator-ui.md](../reference/operator-ui.md) "Projects +
engine restart" and [../reference/render-engine.md](../reference/render-engine.md) §1b.

### A2 — Clip grid (D12) — built 2026-10-10, end-to-end gate open

Shipped: region × clip-slot grid (full in Prepare's drawer, compact in
Perform), click = bind on DESIGN via `sceneCommit` per C5, drag re-file
(app-owned `clipgrid.json`), NEW flags + "new from agent" lane, 2 s
polling. Verified against fixture HAP clips (C2/C3) on the engine working
tree: a dropped clip showed as NEW within one poll, one click + T put it on
the wall inside its region; swapping a region's clip in Perform is one
click + T. See [../reference/operator-ui.md](../reference/operator-ui.md)
"Clip grid".

**Still open — the real end-to-end gate:** agent render → on the wall in
< 2 min, hands on design only. HAP decode (E1b) landed after the grid
(`d03e78e`); run the gate once **E1c** (region placement from
`source_box`; in progress on 2026-10-10) and **P5** (`to_hap` +
thumbnails written by the MCP) land: one `texture_flow(region=…)` render
from an external Claude session, then click + T. Fixtures:
`uv run python wzrd-app/scripts/fixture_projects.py`.

### F1 — Seed clip effects (D14)

FX (a WGSL-authoring agent with the engine running) · needs E1 step b ·
`library/effects/` (C6), copied into the gig project's `effects/`. The ten
effects in D14, each a clip effect: declares the `image` input and reads it
through `sample_image` with its own uv/colour work. Self-contained WGSL — the
`#import` helpers are post-gig, and toolbox phase 2 re-cuts these as chain
filters (a mechanical `sample_image` → `sample_prev` swap). **Gate per
effect:** probe green, declared min/max on every dial, seven clips + the
effect on each at 60 fps on the M2, and `check_frames`'s flash-rate check
passes at default dials (strobe gate included — its default must be safe
for a lying-down audience).

### P7 — Style recipes

PY · needs P3 · `library/recipes/` (C6): one folder per recipe —
`recipe.json` (TextureFlow args, prompt, notes, when to use), style images,
`preview.jpg`. Hand-curated and agent-saved recipes share the folder. Tools:
`list_recipes` (compact table), `save_recipe(from_clip=…)` (from the clip's
sidecar), a `recipe` arg on `texture_flow`. **Gate:** the agent builds a
20-clip library from one recipe set.

### S1 — Skills

SKILL · needs everything above that it names · `wzrd_mcp/skills/`. A new
**on-site run sheet** workflow skill (photos → canvas → regions → pack →
library from recipes → show checklist: warm GPU, flash/haze checks); the
`textureflow` skill points at recipes, `region=` and `check_frames`; the
`live-engine` skill teaches C5 clip bindings and the `image` input.
Also teach the **region story** flow (Xander, 2026-10-10): a story on one
oddly shaped region (a tall pillar) = keyframes from that region's
`region_surface_image` with `aspect_ratio="auto"`, interpolated with the
video model at its nearest aspect, `region=` on every call, then placed in
the region's `source_box` (engine, C5 contain-fit) or `reproject_video`.
The plumbing lands with P1/P3; only the skill text is missing.
**Gate:** a cold Claude session, given only the venue photos, follows the
run sheet to a playable project without being told tool names.

### Optional — MIDI controller

APP, after A2 · only if weeks 1–3 landed on time. Spec:
[midi-controller.md](midi-controller.md). First to go if anything slips.

## If something slips, cut in this order

MIDI, the recipe tools (the agent can read a recipe folder without
dedicated tools), seed effects beyond the first four, the probation window.
(The project switcher and the "new from agent" strip already landed.) The
engine video path, the aspect fix, the starter scene and the clip grid
can't slip.

## After the gig

Post-gig build order (effects library, effect chains + feedback + post, TAKE
transitions, in-app generation, a setup room, MIDI): [vj-toolbox.md](vj-toolbox.md).

## Parked (revisit after the gig, with a reason to)

- `stabilize_to_surface` — the drift it targeted is mostly the D8 bug.
- `composite_videos` — the engine composites regions live (D4).
- `edit_video` — TextureFlow output loops by default; shaders handle the rest.
- `interpolate_video` / `upscale_video` — TextureFlow ends in a FILM/RIFE node.
- Text-prompted segmentation (SAM-class), `reconcile_layerpack` (stable layer
  ids across re-segmentation), `surface_report`. (`list_assets` /
  `mark_asset` are unparked into [vj-toolbox.md](vj-toolbox.md) phase 4.)
- Projector-mounted camera ("eyes"; the disabled `capture_camera_snapshot`
  tool exists), multi-projector output (base M2 drives one external display),
  in-app chat, audio/beat analysis, H.264 playback.
