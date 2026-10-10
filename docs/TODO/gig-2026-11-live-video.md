# Gig build plan — live region video + agent content tools (≈ 21 Nov 2026)

> **Status: agreed build plan, nothing built yet.** Settled with Xander in a
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

**C1 — `project.json`** (project root, written by the MCP, owner P1):
`{ "schema": 1, "canvas": [1376, 768], "spend_cap_usd": 20 }`. `canvas` is
set once by `set_project(canvas=…)` and is immutable afterwards (a different
value = error naming the current one). Absent file = legacy project: tools
behave as today, minus the D8 stretch.

**C2 — provenance sidecar** `<output>.meta.json` next to every file a tool
writes (owner P3): `{ "schema": 1, "tool", "args" (incl. prompt / refs /
seed / model), "inputs": [paths], "parent": path|null, "created": ISO,
"est_usd": float, "region": layer id|null, "source_box": [x0,y0,x1,y1]|null }`.
Reserved for [vj-toolbox.md](vj-toolbox.md) phase 4: `"rating":
"favourite"|"reject"`. Readers ignore unknown keys.

**C3 — clip files** (owner P5): a playable clip is a HAP `.mov` under the
project's `generated/`, stem shared with its source: `<stem>.mp4` (source),
`<stem>.mov` (HAP), `<stem>.mov.meta.json` (its own sidecar, `parent` = the
mp4, `region`/`source_box` copied, `est_usd` 0), `<stem>.thumb.jpg` (320 px
wide, middle frame). **The MCP writes the thumbnail at transcode time** — the
app has no MCP client before the toolbox, so it never generates one; a clip
without a thumb shows a placeholder.

**C4 — `source_box` in `pack.json`** (owner P4, read by E1): optional
per-layer `"source_box": [x0, y0, x1, y1]`, pixel space of the masks,
max-exclusive — the same convention as the existing `bbox`. Absent → the
engine falls back to `bbox`.

**C5 — clip binding in `scene.json`** (owner E1, written by A2 and the agent):

```jsonc
{ "id": "clip_fabric_left",
  "select": { "id": "fabric_left" },
  "effect": "videoClip",                 // or any effect declaring an `image` input
  "params": {
    "image": { "clip": "generated/tf_0412.mov",   // relative to the scene dir
               "placement": "region",             // "region" (default) | "canvas"
               "rate": 1.0 } } }
```

- Descriptor input `{ "name": "image", "type": "image" }`; at most one per
  effect in v1. WGSL reads it with the prelude accessor **`sample_image(uv)`**
  (canvas uv in, premultiplied RGBA out; transparent outside the region's box
  under `"region"`). A binding that resolves to N layers places the clip in
  each layer's own box.
- **Playhead is stateless:** `(leg transport time × rate) mod duration` (HAP
  is all-intra, so any frame is a free seek). Promote already copies
  transport, so a clip continues across TAKE without a jump, and the speed
  master bends clips for free. E1 confirms in its design pass; if it
  changes, it changes here first.
- Missing file, non-HAP file or no BC support → that binding draws nothing
  and reports a load error; every other binding keeps playing.

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

### P1 — Locked canvas + aspect fix (D7, D8)

PY · needs nothing · `wzrd/utils.py`, `wzrd_mcp/project.py`,
`wzrd_mcp/project_tools.py`, the image tools that size outputs.
`set_project` gains a one-time `canvas` (C1). Remove the
tolerance-stretch in `utils.py`; undo Nano Banana's crop exactly. Tools
conform to / check against the canvas; `build_layerpack`'s
`projector_resolution` defaults to the canvas. **Gate:** a fabric seg map's
masks land within 1–2 px of the surface; re-running the drift measurement
above shows < 0.5 px keyframe alignment.

### P2 — `import_photos` + EXIF orientation

PY · needs P1 · new tool + the shared input loader. HEIC via `pillow-heif`,
EXIF orientation, from `~/Downloads` by recency → project `captures/`,
conformed to the canvas; EXIF orientation applied in the shared loader for
every tool. **Gate:** an iPhone HEIC shot in portrait lands upright and at
canvas size.

### P3 — Provenance sidecars + spend cap

PY · needs P1 (`project.json`) · `wzrd_mcp/_log.py` (`logged_tool`), the
generation tools. `logged_tool` writes a C2 sidecar for every local output
path in a result. Add `est_usd` for Nano Banana, Kling and TextureFlow.
`texture_flow`, `kling_*` and `nano_banana` gain an optional `region=<layer
id>`: recorded in the sidecar together with that layer's `source_box` from
the project's `pack.json`. Spend = sum of `est_usd` over the project's
sidecars; generation tools warn at 80 % of `spend_cap_usd` and refuse past
it unless called with `allow_over_cap=True` (the agent asks Xander first).
No `list_assets` / `mark_asset` yet. **Gate:** a TextureFlow render with
`region=` produces a sidecar naming region + box; a call past the cap is
refused with the running total.

### P4 — `source_box` in the pack + starter `scene.json`

PY · needs nothing · `wzrd/layerpack.py`, `build_layerpack` in
`wzrd_mcp/tools.py`. `build_layerpack` reads `islands.json` (written next to
the masks by `extract_color_regions`) and writes each layer's padded box as
C4. If the project has no `scene.json`, write a starter one (`pack` →
`packs/<pack>`, `transport.bpm` 120, empty `bindings`/`post`) — never
overwrite an existing one. Reason: the engine refuses to boot without a
scene + pack (`Core::new`), and nothing else writes a scene offline, so a
fresh venue project could not be opened by the switcher (absorbs the old
backlog `build_scene` item). **Gate:** fresh project → seg map → regions →
`build_layerpack` → `cargo run -- --scene <project>/scene.json` boots black
with the pack loaded; every layer's `source_box` matches `islands.json`.

### P5 — Clip pipeline: `to_hap`, thumbnails, `extract_frames`

PY · needs P3 (sidecar inheritance) · `wzrd_mcp/media_tools.py`,
`texture_flow` / `kling_*` download path. `to_hap(video, codec="hap"|"hap_q")`
via local ffmpeg → C3 files; pad dimensions to multiples of 4 (BC works in
4×4 blocks; ffmpeg's `hap` encoder refuses otherwise), keep fps. TextureFlow
and Kling run it automatically after download (`hap=True` default).
`extract_frames(video, at=[…|"first"|"last"], count=None)` → stills.
**Gate:** one TextureFlow render arrives as mp4 + HAP `.mov` + both
sidecars + thumb; the `.mov` loops in `ffplay` without a seam jump.

### P6 — `check_frames`

PY · needs nothing · new numeric module in `wzrd/`, thin wrapper in
`wzrd_mcp/media_tools.py`. Video/projector checks only: loop-seam score
(last → first), **photosensitive flash rate** (> 3 large flashes/s = fail; the
audience lies staring at the ceiling), dark-floor haze, lit fraction, and the
still-vs-surface shift check gated at ≈ 0.1 %. Compact JSON, no pixels to
the LLM. **Gate:** a synthetic 5 Hz strobe fails, a TextureFlow loop passes,
a keyframe shifted 3 px fails the shift check.

### E1 — Engine video: HAP clips as an effect `image` input (roadmap §5.9)

ENG · needs C4/C5 only · new `render-core/src/video.rs`, effect registry +
prelude, `gpu.rs`, compositor; step d touches `wzrd_mcp/engine_tools.py`.
Read roadmap §5.9 (design + the staging-ring invariant) and reference
§1b/§2.1/§2.4/§2.6 first. Four steps, each its own commit and gate:

- **a. The `image` input.** Descriptor type `image`, the `sample_image(uv)`
  accessor, built-in `videoClip`, canvas placement — fed from a still PNG
  first so the binding layout is proven before any decoding. Gate: headless
  scene shows a PNG cut by a mask; bad path → load error, rest plays.
- **b. HAP decode.** MOV demux + snappy + BC upload on a decode thread
  through the staging ring; `TEXTURE_COMPRESSION_BC` requested only if
  offered; the C5 stateless playhead. Plain HAP (BC1) required; HAP Q (BC3
  YCoCg, needs a colour convert) only if cheap; no alpha variant (additive:
  black is already dark). Gate: seven clips at 60 fps (p95) on the M2, **and
  a TAKE crossfade that swaps all seven clips holds 60 fps** (both legs
  decode during the fade).
- **c. Region placement** from `source_box`, `bbox` fallback. Gate: **six
  fabric clips + a ceiling clip looping at 60 fps on the M2**, each inside
  its own region.
- **d. Agent + docs.** `get_scene_context` shows clip params,
  `upsert_binding` documents C5; reference §2.1/§2.4/§1b updated, BC
  exception added to CLAUDE.md. Gate: an external Claude session binds a clip
  to a named region on design from a plain-language ask.

### E2 — Post-swap probation window (D15, roadmap §5.11)

ENG · needs E1 (same files, serial) · after any pipeline/plan swap on
**live**, watch ~30 frames; if p95 blows the budget and the swap is the
delta, auto-revert to the retained previous pipeline and emit `hot_reload`
`{reverted: "frame budget"}`. Trap: a TAKE crossfade (both legs rendering,
clip streams opening) legitimately raises frame time — attribute or exclude
it, or every big TAKE reverts itself. **Gate:** a deliberately slow shader
promoted to live reverts within ~0.5 s; a one-hour set of normal TAKEs
triggers zero reverts.

### A1 — Project switcher + "restart engine" (D13)

APP · needs nothing (P4 makes new projects openable) ·
`src-tauri/src/engine.rs`, `lib.rs`, TopBar. Tear down and respawn the
render thread on a new scene path (same teardown as
`RunEvent::ExitRequested`); the webview re-hydrates from scratch (store
reset, `lockLegs`). The list = folders under the projects root
(`WZRD_PROJECTS_ROOT`, default `<repo>/projects`) that contain a
`scene.json`. Recent list in a per-machine `settings.json` in the Tauri app
config dir (MIDI's machine default goes there later). **Gate:** switch
between two projects mid-session, each comes back with its own alignment,
masters and looks; engine crash → one click restores.

### A2 — Clip grid (D12)

APP · needs C3/C5 to start, E1 + P5 to finish · new grid components,
`src-tauri/src/rpc.rs` + the dev file middleware (browser bridge twin).
A `clips_list` command returns the project's HAP clips with sidecar + thumb
path; new files appear without a reload (poll or watch). Rows = the pack's
regions (label + mask glyph from `state/masks.ts`) plus "unassigned"; slots
newest first, placed by the sidecar's `region`; drag re-files a clip onto
another row. Clicking a slot commits through `sceneCommit.ts` → design: set
`image.clip` on the region's first binding whose effect declares an `image`
input, else create `clip_<region>` with `videoClip` (C5). Active slot per
row highlighted per leg (design vs live colour). Fresh clips flagged **NEW**
until first clicked (in memory), plus a **"new from agent"** strip (also in
Perform). Full grid in Prepare, compact one under the layer stack in
Perform. Looks + auto-pilot need no change (a clip is just an effect input
inside a look). **Gate:** agent render → on the wall in < 2 min, hands on
design only; swapping one region's clip in Perform is one click + T.

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
**Gate:** a cold Claude session, given only the venue photos, follows the
run sheet to a playable project without being told tool names.

### Optional — MIDI controller

APP, after A2 · only if weeks 1–3 landed on time. Spec:
[midi-controller.md](midi-controller.md). First to go if anything slips.

## If something slips, cut in this order

MIDI, the "new from agent" strip (NEW flags in the grid still show fresh
renders), the recipe tools (the agent can read a recipe folder without
dedicated tools), seed effects beyond the first four, the probation window,
then the project switcher (fallback: relaunch with `WZRD_SCENE`). The engine
video path, the aspect fix, the starter scene and the clip grid can't slip.

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
