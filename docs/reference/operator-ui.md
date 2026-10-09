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
| **Prepare** | Build looks on DESIGN: layer list + region list, the surface map (photo / light / both), inspector (target, effect cards, dials), code drawer (Monaco, lazy). Entering it switches hands to DESIGN. Two-click layer creation: pick a region → "+ New"/"Light it up". | `Prepare.tsx`, `SurfaceMap.tsx`, `LayerParams.tsx mode="author"`, `MonacoPanel.tsx` |
| **Align** | §2.8 warp; side panel with the "projector bumped?" recovery hint, test patterns, background (lit-background warning). | `Align.tsx`, `WarpCanvas.tsx`, `state/alignment.ts` |
| **Perform** | The booth: monitor (native preview, deck-coloured frame, PREVIEW/PROGRAM label), looks bank + auto-pilot, layer stack (mixer), selected layer's dials, deck panel (switch, cue monitor, TAKE/CUT/PULL, FADE, BAR/NOW), masters + BLACK/FLASH/FREEZE + room audio. | `Perform.tsx`, `DeckPanel.tsx`, `LayerStack.tsx`, `LayerParams.tsx`, `LooksBank.tsx`, `MastersPanel.tsx` |
| **Debug** | Engine internals (unchanged content, restyled). | `Debug.tsx` |

`TopBar.tsx` is shared: rooms, compact DESIGN|LIVE + TAKE outside Perform,
tempo (beat lamps from the `transport` channel, BPM double-click/scroll,
TAP, `1` = bar line here), health (Audio / FPS / reload verdict / Engine),
ADOPT for agent edits, help overlay (`?`), toasts.

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
`hello {actor:"ui"}`) plus a dev-only Vite middleware (`vite.config.ts`,
`/__wzrd/<cmd>`) for the shell's file commands, bound to `WZRD_SCENE`. Use
it to drive and screenshot the UI with browser tooling while `pnpm tauri
dev` runs. Two seats are independent: edits from one aren't mirrored into
the other's store (both declare actor `ui`).

## Traps

- **No emitted `vite.config.js`.** Vite resolves `.js` before `.ts`; a
  `tsc -b`-emitted copy silently pinned a stale config (found 2026-10-09).
  `tsconfig.node.json` now emits to `node_modules/.tmp/`.
- **Monaco is bundled, not CDN-loaded** (`monacoSetup.ts`: editor core +
  JSON only, local workers, `wzrd` theme). The CDN default rendered a blank
  editor offline — venues are often offline.
- `pnpm add` under a running dev server breaks Vite's resolved paths;
  restart `pnpm tauri dev`.
- Colour defaults from `effect.describe` arrive as float arrays — convert
  with `LayerParams.tsx::toHex` before writing a scene.

## Keyboard (single keys, ignored while typing)

D / L hands · T take · ⇧T cut · P pull · B blackout · F (hold) flash · Z
freeze · Tab tap · `` ` `` bar line · ↑↓ / 1–9 select layer · M / S mute /
solo · [ ] opacity · ⇧1–9 cue look (⇧⌥ = GO) · ⌘S save design · ? help.
Faders: press = jump, ⇧-drag = fine, double-click = reset, wheel = nudge.
