# wzrd-app — Phase 4 Tauri shell

The control surface that wraps `render-core` for humans. Three routes —
**Prepare** (surface canvas + Monaco editor + binding inspector), **Perform**
(preview hero + audio strip + driver rack), **Debug** (collapsible panels).
The headless agent path (`render-core --scene foo.json`) is unchanged; this
shell is purely additive.

## Architecture

```
┌──────────────── wzrd-app process (Tauri) ───────────────┐
│                                                         │
│  React + TS webview ◀──Tauri invoke──▶  Rust commands   │
│  (3 routes, Monaco)                     (src-tauri/rpc) │
│                                                         │
│                       │                                 │
│                       │ JSON-RPC over WebSocket         │
│                       ▼ ws://127.0.0.1:9123             │
└─────────────────────────────────────────────────────────┘
                        │
              ┌─────────▼─────────┐
              │  render-core      │   ← subprocess (sibling
              │  (spawned by      │     of audio server in
              │   the shell)      │     the §3.10 sense)
              └───────────────────┘
                        │
                        ▼
                projector display
```

Why a subprocess and not in-process winit? See decision P4 in
`../docs/reference/render-engine.md` — short version: it sidesteps the macOS
NSApp main-thread fight and reuses the *exact* RPC surface Phase 7 needs for
MCP, with no shim code in between. (The analyzed-but-uncommitted single-process
alternative lives in `../docs/TODO/single-process-collapse.md`.)

## Dev workflow

Two terminals during development:

```bash
# 1. Build the engine so the Tauri shell can spawn it
(cd ../render-core && cargo build)

# 2. Build + run the Tauri app
cd wzrd-app
pnpm install
WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json \
  pnpm tauri dev
```

The shell finds the engine binary by:
  1. `WZRD_ENGINE_EXE` env var (override),
  2. `target/{debug,release}/render-core` next to the Tauri executable,
  3. `../../render-core/target/debug/render-core` (workspace dev layout).

The scene path comes from `WZRD_SCENE` env var or `--scene <path>` passed
through to `wzrd-app`.

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
  dropdowns.
- **Perform**: large preview + audio band/onset visualizer + driver rack
  listing every driver-bound param in the active scene with live values.
- **Debug**: connectivity, render-stats, driver-bus snapshot, hot-reload
  events history, log stream with level filter, and a raw pack/scene dump.

Routes switch with `⌘1` / `⌘2` / `⌘3` (Ctrl on non-mac).

## Telemetry channels

Subscribed by the Tauri shell on connect; relayed onto the webview's
`engine:telemetry` event channel.

| channel | rate | consumer |
|---|---|---|
| `preview` | ~15 fps | corner thumbnail / Perform hero |
| `hot_reload` | event | status strip + Debug history |
| `audio_freshness` | 1 Hz heartbeat + transitions | OSC pill |
| `fps` | 2 Hz | FPS pill + Debug render-stats |
| `frame_stats` | 2 Hz | Debug render-stats |
| `audio` | 30 Hz (when present) | Perform audio strip |
| `drivers` | 30 Hz (when present) | Perform driver rack + Debug |
| `connectivity` | ad-hoc | Debug connectivity |
| `log` | ad-hoc | Debug log stream |

Sticky channels (`hot_reload`, `audio_freshness`, `connectivity`, `fps`)
are replayed to new subscribers so a freshly-opened webview shows the
right pills immediately.
