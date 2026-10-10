# Operator UI — the `wzrd-app` webview

> How the human-facing control surface is organised and why. Engine
> contracts it talks to (RPC, telemetry, legs, alignment) live in
> [render-engine.md](render-engine.md) — §2.x pointers below go there. The
> north star is [user_design_spec.md](user_design_spec.md). Redesigned
> 2026-10-09 (Resolume-style revamp); the previous rack/inspector UI is gone.

## The three rules every control obeys (`src/state/actions.ts`)

1. **Hands go where the deck points.** The DESIGN | LIVE switch
   (`deck.preview_source`, §2.6) is the *control leg*: layer faders,
   mute/solo and knob pins write that leg. The whole room re-tints with it —
   `<html data-deck>` flips the `--deck` CSS token (cyan DESIGN, red LIVE),
   so the mode is visible from across the booth.
2. **Masters are the room, not a leg.** `setMaster` writes **both** legs;
   `App.tsx::lockLegs` copies live's masters onto design and calls
   `transport.set {sync}` at hydrate. With identical speed masters the two
   clocks then stay in lockstep, so a TAKE (which copies design's control
   state + clock into live) never jumps the wall's brightness, speed or bar
   grid. Don't reintroduce per-leg master writes.
3. **Structure is authored in DESIGN.** Effect choice, region targeting,
   colours, modulation routing/ranges are scene edits → `sceneCommit.ts` →
   design leg (engine blanket rule). On LIVE those controls render read-only
   with an "edit in DESIGN" affordance; scalar faders stay playable as pins.

Live writes are **throttled (leading + trailing), never debounced**
(`actions.ts::throttled`, 25 ms). The old rack used a 25 ms *debounce*,
which never fired mid-drag (pointer events arrive every ~16 ms) — faders only
"landed" when the hand paused. That was half of "the sliders don't do much";
the other half was guessed ranges (e.g. `scale: 50` on a 0.5–8 control).
Every scalar now uses the effect descriptor's `min`/`max` (`LayerParams.tsx
::rangeFor`, extended to include an out-of-range scene value).

## Rooms (`src/routes/`, ⌘1–⌘3, Debug ⌘4 / ⚙)

| Room | What it's for | Key files |
|---|---|---|
| **Prepare** | Build looks on DESIGN: layer list + region list, the surface map (photo / light / both), inspector (target, effect cards, dials), bottom drawer = **clip grid** or code (Monaco, lazy). Entering it switches hands to DESIGN. Two-click layer creation: pick a region → "+ New"/"Light it up". | `Prepare.tsx`, `SurfaceMap.tsx`, `ClipGrid.tsx`, `LayerParams.tsx mode="author"`, `MonacoPanel.tsx` |
| **Align** | §2.8 warp; side panel with the "projector bumped?" recovery hint, test patterns, background (lit-background warning). | `Align.tsx`, `WarpCanvas.tsx`, `state/alignment.ts` |
| **Perform** | The booth: monitor (native preview, deck-coloured frame, PREVIEW/PROGRAM label), looks bank + auto-pilot, layer stack (mixer) with the compact clip grid under it, selected layer's dials, deck panel (switch, cue monitor, TAKE/CUT/PULL, FADE, BAR/NOW), masters + BLACK/FLASH/FREEZE + room audio. | `Perform.tsx`, `DeckPanel.tsx`, `LayerStack.tsx`, `ClipGrid.tsx compact`, `LayerParams.tsx`, `LooksBank.tsx`, `MastersPanel.tsx` |
| **Debug** | Engine internals (unchanged content, restyled). | `Debug.tsx` |

`TopBar.tsx` is shared: project menu (`ProjectMenu.tsx`), rooms, compact DESIGN|LIVE + TAKE outside Perform,
tempo (beat lamps from the `transport` channel, BPM double-click/scroll,
TAP, `1` = bar line here), health (Audio / FPS / reload verdict / Engine —
a restart button while the engine is down), ADOPT for agent edits, help
overlay (`?`), toasts. While the engine is down a red banner under the bar
(`EngineDownBanner`) shows the reason and one "Restart engine" button.

## Projects + engine restart (gig A1)

A **project** is a folder with a `scene.json` under the projects root
(`WZRD_PROJECTS_ROOT`, else `projects_root` in settings, else
`<repo>/projects` — the MCP's default). The engine's scene dir is fixed at
boot, so **opening a project re-roots the in-process engine** (shutdown +
boot on the same windows, ~150–250 ms; host side in
[render-engine.md](render-engine.md) §1b "Re-rooting"). Each project brings
its own pack, effects, looks, `alignment.json` and `session.json`
(masters, knobs) back. "Restart engine" is the same move on the current
scene — the crash recovery.

- Shell: `src-tauri/src/projects.rs` (`projects_list`, `project_current`,
  `project_open`, `engine_restart`). A failed open re-opens the previous
  project and returns the error; a failed boot leaves the shell up with
  the reason (`engine_status.last_error`) instead of dying.
- **Per-machine `settings.json`** in the Tauri app config dir
  (`~/Library/Application Support/build.gigabrain.wzrd/`): `recent`
  (scene paths + `opened_ms`), optional `projects_root`; unknown keys are
  preserved (MIDI's machine default lands here later). With no
  `WZRD_SCENE` / `--scene` the app boots the most recent project — a
  Finder launch works.
- Webview: opening takes two clicks (arm → "OPEN", the wall goes dark for
  a moment). `actions.ts::openProject` first **settles** in-flight scene
  commits into the old project (`sceneCommit.ts::settleCommits`); every
  hydrate then starts from `store.resetProject()` + `dropPendingCommits()`
  + `resetActionState()`, so nothing of the previous engine leaks in.
  Hydrate fires on the engine coming back (status edge) and on the
  shell's `engine:project` event, deduped (`App.tsx::rehydrate`).
- Test fixtures: `wzrd-app/scripts/fixture_projects.py` builds
  `projects/fixture-a` + `fixture-b` (example pack, real HAP clips with
  C2/C3 sidecars + thumbs); `--add-clip <project> --region <id>` simulates
  an agent render landing.

## Clip grid (gig A2)

Resolume's layer rows, but each row is a **region** of the pack (mask
glyph + name) plus "Unassigned". Slots are the project's HAP clips
(gig plan contract C3: `generated/<stem>.mov` + `<stem>.thumb.jpg`),
newest first, placed by the C2 provenance sidecar's `region`.

- **Click a slot = play that clip on that region in DESIGN**
  (`state/clips.ts::bindClip`, through `commitSceneMutation`): set
  `params.image.clip` (C5) on the first binding whose `select.id` is that
  region and whose effect declares an `image` input (keeps its
  placement/rate), else append `clip_<region>` with the built-in
  `videoClip`. TAKE as usual — in Perform a swap is one click + T.
- Design's clip per region gets a design-colour ring, live's a red top
  bar (`clipsByRegion` over both legs' scene JSON; only exact `select.id`
  bindings count — a tag/group binding isn't "this region's clip").
- **Drag** a slot onto another row to re-file it. The filing is app state
  in `<project>/clipgrid.json` (`{"schema":1,"rows":{clip: region|null}}`);
  the sidecar is never touched — it keeps saying what the clip was made
  for. An unassigned clip can't be clicked until it's filed.
- **NEW** = arrived after the grid's first load for this project (in
  memory, cleared on click). NEW clips also line up in a "new from agent"
  lane on top of the grid (both rooms).
- Shell: `src-tauri/src/clips.rs` — `clips_list` (dir scan + sidecars,
  incl. the source mp4's sidecar via `parent` for prompt/tool/cost, plus
  the filing map), `clip_thumb` (base64; the webview has no asset
  protocol), `clip_file`. Polled every 2 s from `App` (`useClipsPoller`),
  so NEW is right in every room.

## Mechanisms worth knowing

- **Layer mixer = reserved overrides.** Opacity / mute / solo are §5.5
  overrides under `@opacity` / `@mute` / `@solo` on the binding (engine
  folds them into the per-pass `layer.opacity`, render-engine.md §2.5). So
  they're per leg, zero-rebuild, persisted in `session.json`, copied by
  TAKE/PULL. The store mirrors both legs' override tables
  (`store.overrides`, hydrated from `param.list`, patched optimistically by
  `actions.pin`). Solo click is exclusive, ⇧-click additive.
- **Pins vs scene values.** In Perform, dragging a scalar *pins* it on the
  control leg (amber `PINNED`, tick at the scene value, double-click
  releases). In Prepare (`mode="author"`) the same fader writes scene.json.
  `bakeDesignPins` (⌘S, and every Save look) writes design pins into the
  scene and clears them.
- **Modulation** is a driver with `range: [from, to]` (render-engine.md
  §2.1). `MOD_SOURCES` in `actions.ts` maps VJ words (Kick/Snare/Hats =
  `audio.onset` low/mid/high, Bass/Mids/Highs = `audio.band`, Beat/Bar/4
  Bars/Slow = clocks) to driver specs; `RangeFader` edits rest→peak, with
  the live value from the `drivers` channel riding inside.
- **TAKE arming** compares design vs live scene JSON (`designDiffers`,
  key-sorted, ignoring `transport` because TAP moves both legs live).
  Live JSON is re-read after every ramp completes (`deck` channel → idle).
- **Looks** are plain scene files in `<scene_dir>/looks/<name>.json` + a
  `.jpg` thumbnail (Tauri commands `looks_list/look_read/look_write/
  look_delete` in `src-tauri/src/rpc.rs`). Click = cue into DESIGN, double
  click / ▶ = cue + TAKE (`recallLook` awaits the design apply first via
  `commitSceneNow`, so GO never fades to the previous design).
- **Auto-pilot** (`state/autopilot.ts`) walks the looks bank every N bars:
  recall 2 beats before the boundary, bar-quantized TAKE lands on the line.
  UI-driven stopgap; it stops when the app closes (the engine playlist in
  the roadmap §5.6 supersedes it).
- **Shared mask assets** (`state/masks.ts`): each region's mask is loaded
  once → tinted overlay, 256-px pick map, silhouette thumbnail (the region
  glyphs in every list) + the pack's `surface.png` for the map's photo mode.
- **The monitor can't be drawn on.** The native preview is a child window
  over the measured slot (`NativePreview.tsx`, `useAspectFit`); all chrome
  sits around it. In a browser it falls back to the JPEG `preview` channel.

## Browser bridge (dev harness)

The same Vite build runs in a plain browser: `src/api/transport.ts` picks
Tauri `invoke`/`listen` inside the shell, else `wsBridge.ts` — JSON-RPC to
the engine's WS (`ws://127.0.0.1:9123`, `?ws=` overrides; declares
`hello {actor:"ui"}`) plus, for everything the shell does itself (files,
looks, projects, clips, engine restart), the shell's **debug-build-only
HTTP twin** `src-tauri/src/devport.rs` on `127.0.0.1:9124`
(`WZRD_DEVPORT`), reached through Vite's `/__shell` proxy. It calls the
same Rust command functions on the main thread, so a browser seat follows
project switches (the old Vite file middleware was pinned to
`WZRD_SCENE`). A switch shows up in the browser as an engine-socket
reconnect → hydrate. Use it to drive and screenshot the UI with browser
tooling while `pnpm tauri dev` runs. Two seats are independent: edits from
one aren't mirrored into the other's store (both declare actor `ui`).
Browser-tooling traps (Claude in Chrome): CDP's synthetic mouse drag
doesn't fire HTML5 drag-and-drop — exercise the clip grid's drop handlers
with dispatched `DragEvent`s; and the screenshot's coordinate frame can
differ from the page's CSS viewport (seen: 1456 vs 1543 px), so
coordinate clicks miss — click via element refs (`find`). In dev, the
modules are importable from the page (`await import('/src/state/store.ts')`)
to read store state as evidence.

## Traps

- **No emitted `vite.config.js`.** Vite resolves `.js` before `.ts`; a
  `tsc -b`-emitted copy silently pinned a stale config (found 2026-10-09).
  `tsconfig.node.json` now emits to `node_modules/.tmp/`.
- **Monaco is bundled, not CDN-loaded** (`monacoSetup.ts`: editor core +
  JSON only, local workers, `wzrd` theme). The CDN default rendered a blank
  editor offline — venues are often offline.
- `pnpm add` under a running dev server breaks Vite's resolved paths;
  restart `pnpm tauri dev`.
- A Rust edit makes `tauri dev` restart the app before the old process
  has let go of its ports: the dev port retries its bind, but the engine
  WS (9123) can lose the race — the app then comes up with the engine
  down and the banner's "Restart engine" fixes it.
- Colour defaults from `effect.describe` arrive as float arrays — convert
  with `LayerParams.tsx::toHex` before writing a scene.

## Keyboard (single keys, ignored while typing)

D / L hands · T take · ⇧T cut · P pull · B blackout · F (hold) flash · Z
freeze · Tab tap · `` ` `` bar line · ↑↓ / 1–9 select layer · M / S mute /
solo · [ ] opacity · ⇧1–9 cue look (⇧⌥ = GO) · ⌘S save design · ? help.
Faders: press = jump, ⇧-drag = fine, double-click = reset, wheel = nudge.
