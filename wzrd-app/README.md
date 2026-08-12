# wzrd-app — single-process control shell

The control surface that wraps `render-core` for humans. Four routes, in
load-in order — **Prepare** (surface canvas + Monaco editor + binding
inspector), **Align** (§2.8 output warp), **Perform** (native preview hero +
§5.6 deck bar + masters + audio strip + driver rack), **Debug** (collapsible
panels). The headless agent path
(`render-core --scene foo.json`, standalone binary) is unchanged; this shell
is purely additive.

Since the §5.6 two-deck landing (2026-07-12) the shell drives a **two-leg**
engine: every edit made here lands on the *design* leg (the projector keeps
playing the live leg untouched), and the Perform deck bar's LIVE⇄DESIGN
toggle is a **full control switch** — it picks which leg the native preview
shows *and* which leg the masters row + driver rack read and drive (each
leg owns its own clock, masters, knobs and overrides, so tuning DESIGN
never touches the show). **PROMOTE** crossfades the projector onto the
design content (bar-quantized by default) and copies the design tuning
live — what you previewed is what the crowd gets. **PULL** resets design
back to live (content + controls). New shaders entering design are pre-flight
probed; verdicts (green/yellow/red + predicted full-res p95) surface in the
Debug hot-reload history, with the A/B thresholds editable in Debug. Full
contract: `../docs/reference/render-engine.md` §2.6.

## Architecture (single-process since 2026-07-12)

```
┌──────────────────── wzrd-app process (Tauri) ────────────────────┐
│                                                                  │
│  React + TS webview ◀─Tauri invoke─▶ Rust commands (src-tauri/   │
│  (3 routes, Monaco)                  rpc.rs) → rpc::dispatch     │
│      + borderless native preview          │      (direct call)   │
│        child window over the hero slot    ▼                      │
│  engine-render thread: render_core::Core                         │
│    engine output window (wgpu) ───────────────▶ projector        │
│    native preview blit (same GPU device/composite)               │
│                                                                  │
│  WS server ws://127.0.0.1:9123 — same §3.11 dispatch, for        │
│  external MCP / remote operator clients only                     │
└──────────────────────────────────────────────────────────────────┘
```

The engine runs **in-process** as a library (app-collapse Steps 2–3;
current-state residue in `../docs/reference/render-engine.md` §1b): the
shell owns the engine output
window and a borderless preview child window, and drives `Core` on a render
thread. Local UI calls skip the WS hop entirely; the WS server remains for
external clients.

## Dev workflow

```bash
cd wzrd-app
pnpm install
WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json \
  pnpm tauri dev
```

No separate engine build step — `render-core` compiles in as a Cargo path
dependency. The scene path comes from `WZRD_SCENE` env var or
`--scene <path>` passed through to `wzrd-app`. `WZRD_DISPLAY=<idx>` puts the
engine window borderless-fullscreen on that monitor;
`WZRD_SPIKE=panic|device_loss` deliberately crashes the render thread ~5 s
in (crash-containment test hook).

To auto-start the audio feature server alongside the shell, set
`WZRD_AUDIO=1` (or pass `--audio` through to the binary:
`pnpm tauri dev -- -- --audio`). The shell runs `uv run audio-server --open`
in `~/Documents/GitHub/Realtime_PyAudio_FFT` (override the checkout location
with `WZRD_AUDIO_DIR`) and kills it on window close. A failed audio spawn is
logged and non-fatal.

## Phase 4.1 deliverable

- Open a pack (read from the scene's `pack` field on engine startup).
- Edit `scene.json` and `effects/*.wgsl` in Monaco with inline naga errors.
- Glance at the status strip during a show (OSC / Engine / FPS / Reload pills).
- Confirm the projector is alive via the corner thumbnail.

Errors never blank the projector: bad WGSL saves → previous good pipeline
keeps rendering, marker surfaces on the Monaco line. Bad `scene.json` →
previous good plan stays active, error toasts on the Reload pill.

## Phase 4.2 deliverable

- **Prepare**: surface canvas with mask overlays, layer hover/select,
  Monaco editor, structured binding inspector with selector/effect/driver
  dropdowns. Highlighting is bidirectional — clicking a region aims the
  inspector at a binding that covers it, and hovering a binding lights up
  every region its selector resolves to.
- **Align**: the §2.8 alignment layer — drag the four corner handles until
  the render lands on the physical surface, drag inside the quad to move the
  whole image, click the outline to drop a handle on an edge, right-click for
  extra handles and their radius. Arrow keys nudge by exactly one output
  pixel (⇧ 10), or move the whole quad when nothing is selected. The grid is
  the real field (drawn from the engine's solved coefficients); the photo
  underlay only follows the corner quad, and the projector is ground truth.
  Test patterns and a background picker for calibrating against a bare wall.
  Persisted by the engine in `alignment.json`; not scene content, so it has
  no save button and never rides along with a scene.
- **Perform**: large preview + audio band/onset visualizer + driver rack
  listing every driver-bound param in the active scene with live values.
- **Debug**: connectivity, render-stats, driver-bus snapshot, hot-reload
  events history, log stream with level filter, and a raw pack/scene dump.

Routes switch with `⌘1` / `⌘2` / `⌘3` / `⌘4` (Ctrl on non-mac).

## Telemetry channels

Subscribed in-process off the engine's telemetry bus; relayed onto the
webview's `engine:telemetry` event channel.

| channel | rate | consumer |
|---|---|---|
| `preview` | ~15 fps | Prepare surface-canvas underlay + Align warp underlay (Perform hero is the native surface, not this channel) |
| `hot_reload` | event | status strip + Debug history |
| `audio_freshness` | 1 Hz heartbeat + transitions | OSC pill |
| `fps` | 2 Hz | FPS pill + Debug render-stats |
| `frame_stats` | 2 Hz | Debug render-stats |
| `audio` | 30 Hz (when present) | Perform audio strip |
| `drivers` | 30 Hz (when present) | Perform driver rack + Debug |
| `connectivity` | ad-hoc | Debug connectivity |
| `log` | ad-hoc | Debug log stream |
| `deck` | transitions + ~10 Hz while a fade ramps | Perform deck bar (promote phase, mix, preview source) |
| `alignment` | boot + every accepted mutation + output resize | Align route (§2.8 document, output size, handle cap) |

Sticky channels (`hot_reload`, `audio_freshness`, `connectivity`, `fps`,
`masters`, `deck`, `alignment`) are replayed to new subscribers so a freshly-opened
webview shows the right pills immediately.
