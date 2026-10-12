# VJ toolbox — from working gig rig to full VJ instrument (post-gig)

> **Status: agreed build order, nothing built yet** (Xander, 2026-10-10).
> Starts **after** the Nov 2026 gig; everything the gig needs is in
> [gig-2026-11-live-video.md](gig-2026-11-live-video.md) — finish that first.
> It also defines things this plan builds on: contracts **C1–C6** (project
> file, provenance sidecar, clip files, `source_box`, clip binding, library
> layout under `library/`) and the gig packages named below. Engine designs
> (§5.x) live in [render-engine-roadmap.md](render-engine-roadmap.md); this
> plan sequences them with their UI, library and MCP work. Current state:
> [../reference/render-engine.md](../reference/render-engine.md),
> [../reference/operator-ui.md](../reference/operator-ui.md),
> [../reference/mcp-server.md](../reference/mcp-server.md).

## Why this plan exists

A gap analysis of "arrive at a venue, set up a full gig from scratch"
(2026-10-10) found the setup chain and the live instrument solid, but:

- **Effects don't travel.** They're project-local; nothing lets you save a
  good one and reuse it at the next venue (design spec §12). There's no
  browsable catalog — only 4 built-ins, the gig's ten clip effects and
  whatever was written per project.
- **Effects can't process.** One effect per binding, no feedback, no global
  post — so most classic VJ effects (trails, kaleido over a clip, glow) are
  inexpressible.
- **Transitions are one crossfade.** TAKE only lerps.
- **Generation and setup need a chat session.** Every MCP tool exists, but
  the app has no buttons for them — "render 4 more for this region" or
  "segment this photo" can't be done from the UI.
- **No hardware controller** (unless it made it into the gig).

The design spec's "not a library of pre-made looks" is about finished
scenes; a library of *effects* is the spec's "shipped effects are starting
points, never ceilings" (spec §3, §12).

## How to hand this off

Each phase ships standalone. Engine first, headless-verifiable, UI second
(roadmap rule) — so most phases are **two packages**: an engine/Python one
that is done when its gate passes headlessly, then a UI one built on it.

| Phase | Packages (lane) | Needs | Parallel with |
|---|---|---|---|
| 0 `#import` | T0 (ENG + FX) | — | 4, 6 |
| 1 Library | T1a (PY + FX), T1b (APP) | T0 | 4, 6 |
| 2 Chains / feedback / post | T2a (ENG), T2b (ENG), T2c (APP) | T0; T2c needs T1b | 4, 6 |
| 3 TAKE transitions | T3a (ENG), T3b (APP) | T2a (`kind` field) | 4, 6 |
| 4 In-app generation | T4a (APP Rust), T4b (APP UI) | — | 0–3 |
| 5 Site room | T5 (APP) | T4a | 0–3 |
| 6 Booth hardening | MIDI (APP), T6 (ENG) | — | anything |
| 7 Agent chat (sketch) | T7 (APP) | Foreman's dogfood checkpoint | anything |

The ENG lane (0 → 1 → 2 → 3) is serial; the APP-only work of phases 4–6
runs alongside it. Watch `src-tauri/src/{lib,rpc}.rs` — T1b, T4a and MIDI
all add commands there; serialize them.

## Phase 0 — shared WGSL helpers (`#import`)

**T0** — roadmap **§5.5 `#import` preprocessor** (`#import noise` resolved
from `<effects_dir>/_lib/` then engine built-ins; real source map for
errors; `_lib/<module>` in the MCP's `effects` facet). Then move the
helpers the gig's ten clip effects duplicate into `library/effects/_lib/`
and re-import them. (The probation window that used to sit here was pulled
into the gig, package E2.) **Gate:** the §5.5 spec's own checks, plus the
ten clip effects compile from shared helpers with unchanged output.

## Phase 1 — global effects library

The gig seeds `library/effects/` (contract C6) with ten clip effects; this
phase makes it a real library.

- **Vendoring, not linking:** projects stay self-contained (a pack must load
  identically next year, spec §7/§12), so the engine never resolves effects
  from the library directly — importing copies `library/effects/<name>/`
  into `<scene_dir>/effects/<name>/`. `_lib/` modules vendor the same way.
- **Library entry** = the project-local effect format (`shader.wgsl` +
  `descriptor.json`) plus `preview.jpg` (the probe thumbnail the engine
  already renders) and descriptor additions: `kind` (source / filter /
  transition, roadmap §5.15), `tags`, a one-line `about`.

**T1a (PY + FX):** MCP tools `list_library_effects` (compact table, filter
by kind/tag), `save_effect_to_library(name)`, `import_effect(name)`; skill
`live-engine` checks the library before writing a new effect from scratch.
Grow the catalog: agent-authored sources (noise fields, gradients,
scanners, particles, caustics), each passing the probe with declared dial
ranges. **T1b (APP):** a Library panel in Prepare (thumbnail grid, kind/tag
filter, search) → drag onto a layer imports + binds on design; "Save to
library" on any effect card.

**Gate:** an effect written at one venue is on a wall at another venue's
project in < 1 min, by drag or by the agent.

## Phase 2 — effect chains, feedback, post

**T2a (ENG):** roadmap **§5.15** per-binding chains (`sample_prev`,
per-stage `@bypass`), including re-cutting the gig's clip effects as
filters (the §5.15 migration note). Decide the execution model — (a)
pooled intermediate textures vs (b) compile-time fusion — at the start;
the roadmap leans (a). **T2b (ENG):** **§5.16** feedback texture + the
consumed `post` chain clipped to the region union. **T2c (APP):** the layer
inspector's effect cards become a vertical **stack** — add from library,
drag to reorder, bypass per card, dials per card; a post strip in Perform
for the global chain. The library grows what these unlock: trails, echo,
smear, glow, blur, slit-scan.

**Gate:** the roadmap gates for §5.15/§5.16, plus building "clip → kaleido →
trails" on a region entirely from the UI.

## Phase 3 — TAKE transitions

**T3a (ENG):** roadmap **§5.17** (transition shaders, region-aware via the
mask atlas; transitions are `kind: "transition"` library entries).
**T3b (APP):** a TRANSITION picker next to FADE in `DeckPanel.tsx`; a look
in the looks bank can carry its own transition, and the auto-pilot uses it.

**Gate:** an auto-pilot run of 6 looks, each landing on the bar with a
different transition, no black frame.

## Phase 4 — in-app generation (the app becomes an MCP client)

The generation tools already exist and are tested; the app just has no
buttons for them. Instead of a new backend, **the app calls the local MCP
server** (`python -m wzrd_mcp`, streamable-HTTP on `:8787`). No LLM in the
app — buttons call tools (roadmap §5.10: Claude Code stays the author).

- **T4a — plumbing (APP Rust):** the shell spawns the MCP server on demand
  like the audio server (`WZRD_AUDIO` pattern in `src-tauri/src/lib.rs`) or
  attaches to a running one; calls go through the **Rust side** (an MCP
  client crate, e.g. `rmcp`) as Tauri commands — not from the webview, which
  would hit CORS and break the browser bridge. Decide first: how
  long-running calls (TextureFlow, Kling: minutes) report progress (MCP
  progress notifications vs watching the output folder for C3 files).
  Unpark `list_assets` / `mark_asset` (PY side): favourite / reject a clip,
  written as the reserved `rating` key of its sidecar (C2).
- **T4b — UI:** **"Render more" per region row** in the gig's clip grid:
  recipe picker (`library/recipes/`), prompt, style images, count, estimated
  cost → `texture_flow` with `region=` (gig package P3); or keyframe → video
  (`nano_banana` + the image→video tool, Kling today). Results land in that
  row flagged NEW (design only, TAKE stays human). **Job tray** (TopBar):
  running/finished jobs, progress, cancel, running spend vs the project's
  `spend_cap_usd` (C1, summed from sidecars). The grid hides rejects.

**Gate:** from the grid, render 4 clips for one region and play one live
without opening a chat session or a terminal.

## Phase 5 — the Site room (daylight setup in the app)

**T5 (APP), needs T4a.** A fifth room (before Prepare in the room order)
that walks the setup chain, each step one MCP call through T4a's client,
each output a plain file in the project so the agent can run the same steps
headlessly (the gig's on-site run sheet skill is the script this room
turns into buttons):

1. **Import** photos (`import_photos`), pick the surface shot, confirm the
   locked canvas (C1).
2. **Prep** (`prepare_surface`), before/after view.
3. **Segment**: seg-map prompt (template from skill `image-prompting`) +
   region hints → `nano_banana` → overlay on the surface; accept or retry.
4. **Extract** (`extract_color_regions`): live preview of regions; sliders
   for `max_colors` / `min_area_fraction` / `delta_e_threshold` re-run it.
5. **Name + group**: rename regions, multi-select → group (roadmap §5.13 UI
   half, built here, reusable from Prepare).
6. **Build** pack + starter scene (`build_layerpack`, which writes both —
   gig package P4), the engine re-roots onto the project (gig package A1),
   hand off to Align.

**Gate:** a new venue photo → aligned, named, grouped regions on the wall in
< 20 min with no chat and no terminal.

## Phase 6 — booth hardening

- **MIDI controller** — **on hold since 2026-10-12** (Xander); full spec in
  [midi-controller.md](midi-controller.md).
  If it landed before the gig, this phase only registers the controls
  phases 1–5 added (transition picker, post chain, render-more) in its
  control registry.
- **T6 — engine-side auto-pilot playlist** (roadmap §5.6 follow-up) so
  auto-pilot survives the app window closing. Ship once promote has been
  proven at the gig.

Both depend on nothing above and can run any time.

## Phase 7 — in-app agent chat (sketch, leans on Foreman)

> **Not designed yet — a direction, not a spec.** Grill it with Xander
> before building. Written 2026-10-10 after reading Foreman.

The gig keeps chat an external Claude Code session (gig plan D11). The
in-app version should **not** sit on phase 4's MCP client: that client is
for agent-less buttons, and the chat's agent has its own MCP connection to
`wzrd_mcp`. Nor should the app embed an LLM loop or its own terminal —
roadmap §5.10 settled that **Claude Code is the author**.

**Foreman** (`~/Documents/GitHub/agent_frameworks/foreman`, Xander's own
tool, read its `AGENTS.md`) solves exactly the hard half: a local Bun daemon
that hosts *unmodified* Claude Code sessions in its own PTY host and gives
each a visual **card** (goal / progress / click-to-answer questions /
handover) plus a proven channel back (human messages batched on Send,
delivered mid-turn via hooks or into an idle TUI only when it reads as
ready; Pause, Stop). Its "page" model — a custom UI the human edits
directly while the agent sees the edits on its next turn — is the same
two-seat shape WZRD already has. So: **lean on Foreman as the agent host;
build only the WZRD-specific panel.** No code sharing (Bun/TS daemon vs
Tauri/Rust; Foreman rules out a desktop shell) and WZRD must not become a
Foreman page (the operator UI is a native app with engine windows).

| Need in WZRD | Foreman primitive to reuse | Already in WZRD |
|---|---|---|
| Run the agent | managed Claude Code session, cwd = project, `wzrd_mcp` loaded | — |
| Agent sees the operator's direct edits without a turn | page-edit diff on next delivery | `changes` ring + actor + `since_rev` (§2.7) — keep it, the engine is the authority, not a file diff |
| "Talk about *this*" | a **tell**: one message + context, sent only from a click | add: a "tell agent" action on layer / effect / clip cards whose context is facet ids + design rev (the `get_scene_context ids` vocabulary) |
| Agent asks ("which of these 4 clips?") | `foreman_ask`: 2–4 options + default + policy | render as buttons in the panel |
| What is it doing | `foreman_brief` / `progress` / `handover` | panel header: goal + now-line |
| Show-night brakes | Stop (one ESC, only while busy), Pause (park after this step) | — |
| Attention | notify only blocking / high-impact | in Perform, a silent badge; never cover the deck |

**Shape, in two steps:** (a) link out — the app shows the bound session's
state + now-line and opens its Foreman card in the browser; (b) a thin chat
panel in the Prepare room (design only — the design spec's "shape of the instrument" puts AI
chat in design) whose Rust side calls
`foremand`'s local API with the CLI bearer token and follows its SSE stream
(not from the webview: Foreman checks exact Origin, same reason as T4a).
Step (b) needs one Foreman-side addition — today its tell route only
accepts page clicks (`pin_id`) — planned in Foreman, not here.

**Preconditions:** Foreman past its own one-week dogfood checkpoint (its
API still moves); unchanged rules — agent works on design only, never
`promote`, ADOPT stays the operator's. If T4a picks "watch the output
folder" for job progress, the job tray shows agent-started renders too.

**Gate:** from the app, ask for "4 more clips for the left fabric, colder",
answer the agent's one clarifying question by click, watch progress, and the
clips land NEW in the grid — no terminal window opened.

## Open questions

- Should the clip grid also hold *effect* slots (Resolume puts generators and
  clips in the same grid)? Decide after the gig, from how the grid felt live.
