# wzrd-app — single-process control shell

The control surface that wraps `render-core` for humans — a Resolume-style
VJ desk. Three rooms in load-in order — **Prepare** (surface map, layers,
effect cards, dials, code drawer), **Align** (§2.8 output warp), **Perform**
(monitor, deck + TAKE, looks bank + auto-pilot, layer mixer, dials,
masters) — plus **Debug**. How it's organised and why:
[`../docs/reference/operator-ui.md`](../docs/reference/operator-ui.md).
The headless agent path
(`render-core --scene foo.json`, standalone binary) is unchanged; this shell
is purely additive.

Since the §5.6 two-deck landing (2026-07-12) the shell drives a **two-leg**
engine: every structural edit made here lands on the *design* leg (the
projector keeps playing the live leg untouched), and the DESIGN | LIVE
switch is a **full control switch** — it picks which leg the native preview
shows *and* which leg the layer faders and knobs drive (the room re-tints:
cyan DESIGN, red LIVE). Masters deliberately write both legs. **TAKE**
(engine verb `promote`) crossfades the projector onto the
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
│      + native preview NSView *under*      │      (direct call)   │
│        the transparent webview            ▼                      │
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
window and a native preview view beneath the (transparent) webview —
`WZRD_PREVIEW=window` falls back to the old preview child window — and
drives `Core` on a render thread. Local UI calls skip the WS hop entirely; the WS server remains for
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
`--scene <path>` passed through to `wzrd-app`; with neither, the app opens
the most recently opened project (per-machine `settings.json` in the app
config dir). Switch projects or restart a crashed engine from the TopBar
project menu — the engine re-roots in-process in a few hundred ms.
`uv run python wzrd-app/scripts/fixture_projects.py` (from the repo root)
builds two test projects with HAP clips for the project menu and the clip
grid. `WZRD_DISPLAY=<idx>` puts the
engine window borderless-fullscreen on that monitor;
`WZRD_SPIKE=panic|device_loss` deliberately crashes the render thread ~5 s
in (crash-containment test hook).

To auto-start the audio feature server alongside the shell, set
`WZRD_AUDIO=1` (or pass `--audio` through to the binary:
`pnpm tauri dev -- -- --audio`). The shell runs `uv run audio-server --open`
in `~/Documents/GitHub/Realtime_PyAudio_FFT` (override the checkout location
with `WZRD_AUDIO_DIR`) and kills it on window close. A failed audio spawn is
logged and non-fatal.

## Driving it from a browser (dev)

With `pnpm tauri dev` running, open `http://localhost:1420` in Chrome: the
same UI talks to the engine's WebSocket plus the shell's debug-only HTTP
port (`src-tauri/src/devport.rs`, 127.0.0.1:9124, behind Vite's `/__shell`
proxy) for files, projects and clips — details in `operator-ui.md`,
"Browser bridge". Handy for
scripted UI testing and screenshots; the native preview is replaced by the
JPEG design feed there.

## Telemetry channels

Subscribed in-process off the engine's telemetry bus; relayed onto the
webview's `engine:telemetry` event as **batches** (an array of frames, at
most one emit per ~33 ms). The streaming channels (`preview`, `audio`,
`transport`, `drivers`, `frame_stats`, `perf`) flow only while a mounted
component demands them (`src/api/telemetryDemand.ts` →
`telemetry_demand`); see `docs/reference/operator-ui.md`.

| channel | rate | consumer |
|---|---|---|
| `preview` | ~15 fps | Perform's cue monitor (sampled ≈1 Hz, full rate on hover); Prepare/Align only in `WZRD_PREVIEW=window` / browser mode |
| `hot_reload` | event | top-bar health + Debug history |
| `audio_freshness` | 1 Hz heartbeat + transitions | top-bar Audio dot |
| `fps` | 2 Hz | top-bar FPS + Debug render-stats |
| `frame_stats` | 2 Hz | Debug render-stats |
| `perf` | 2 Hz | Debug performance readout |
| `audio` | 30 Hz (when present) | Perform room meters + onset lamps |
| `transport` | 30 Hz | top-bar beat lamps + BPM, looks auto-pilot |
| `drivers` | 10 Hz | live values inside the dials (modulated params) + Debug |
| `masters` | on change | Perform masters |
| `connectivity` | ad-hoc | Debug connectivity |
| `log` | ad-hoc | Debug log stream |
| `deck` | transitions + ~10 Hz while a fade ramps | DESIGN/LIVE switch, TAKE progress, room tint |
| `alignment` | boot + every accepted mutation + output resize | Align route (§2.8 document, output size, handle cap) |

Sticky channels (`hot_reload`, `audio_freshness`, `connectivity`, `fps`,
`masters`, `deck`, `alignment`) are replayed to new subscribers so a freshly-opened
webview shows the right pills immediately.
