# render-core

WZRD's realtime additive projection-mapping engine.

Phase 3 deliverable: a playable, audio-reactive engine that runs without
any UI process. Authors and agents express creativity by writing **WGSL**
(inline in `scene.json` or as project-local files) — the engine
hot-reloads on save (D15). Drivers wire clock + audio (OSC ingest from the
external Realtime Audio Feature Server) straight into effect params.
**Still no Tauri, no webview, no TypeScript on the critical path.**

See `../docs/reference/render-engine.md` for the full design (roadmap in
`../docs/TODO/render-engine-roadmap.md`).

## Build

Requires Rust 1.85+ (uses the wgpu 22 / winit 0.30 ecosystem).

```bash
cargo build --release
```

## Run

```bash
# Borderless fullscreen on the primary monitor:
render-core --scene path/to/scene.json

# Pick a specific display:
render-core --scene path/to/scene.json --display 1

# Iterating locally on a laptop — windowed mode:
render-core --scene path/to/scene.json --windowed

# Disable OSC ingest (clocks tick, audio.* drivers return 0):
render-core --scene scene.json --windowed --no-osc

# Bind on all interfaces (audio server on another machine):
render-core --scene scene.json --windowed --osc-addr 0.0.0.0:9000

# Expose the Phase 4 JSON-RPC WebSocket so the Tauri shell (or any other
# §3.11 client) can drive scene / effect / telemetry. Omit for headless:
render-core --scene scene.json --windowed --ws-addr 127.0.0.1:9123
```

The layer-pack path is read from `scene.json`'s `pack` field (resolved
relative to the scene file). Override with `--pack`.

Project-local effects (D15) are picked up automatically from
`<scene_dir>/effects/`. Override with `--effects path/to/effects/`.

## Effects (Phase 3 baseline)

Built-ins shipped as reference implementations — **not the boundary** of
what's expressible:

| name        | inputs                                  | notes                                              |
|-------------|------------------------------------------|----------------------------------------------------|
| `tint`      | `color`                                  | color × mask (debug / baseline)                    |
| `hueCycle`  | `phase`, `color0..color3`                | 4-stop palette cycled by `phase` (often `clock.bars`) |
| `flash`     | `envelope`, `base`, `color`              | additive pulse, typically `envelope = audio.onset` |
| `wobble`    | `amp`, `freq`, `time`, `color`           | UV displacement — edge undulates                   |

User effects live in `effects/<name>/{shader.wgsl, descriptor.json}` or
inline in `scene.json`. Every user effect implements one function:

```wgsl
fn effect(uv: vec2<f32>, mask: f32) -> vec4<f32> {
    // `state` (FrameState) + `layer` (LayerParams) are bound from the prelude.
    // Use f_param(N) for scalar inputs, c_param(N) for colour inputs.
    //
    // Output is *premultiplied* — the compositor blends additively (One+One)
    // so dark pixels stay dark on the projector. Bake the mask (and any
    // intensity scalar) into alpha and multiply rgb by it.
    let a = state.audio_low * mask;
    return vec4<f32>(uv.x * a, uv.y * a, a, a);
}
```

The prelude exposes:

- `state.time`, `state.bar_phase`, `state.beat_phase`, `state.bpm`
- `state.audio_low/mid/high` (post-autoscale into ~[0,1])
- `state.onset_low/mid/high` (decaying envelopes off OSC triggers)
- `state.resolution`
- `sample_mask(uv)` — re-sample the layer's mask at any UV (spatial effects)

## Drivers

Any scalar param can be a literal number OR a driver object:

```jsonc
{ "phase":    { "driver": "clock.bars",  "n": 8 } }
{ "envelope": { "driver": "audio.onset", "band": "low", "decay": 0.18 } }
{ "amp":      { "driver": "audio.band",  "band": "mid" } }
{ "x":        { "driver": "ui.slider",   "name": "x", "default": 0.5 } }
```

Available drivers: `const`, `clock.bars`, `clock.beats`, `clock.phase`,
`clock.time`, `audio.band`, `audio.onset`, `ui.slider` (stub until Phase 4
webview ships). No `audio.rms`, `audio.bpm`, or `audio.fft` in v1 — the
engine exposes exactly what the audio server emits and the scene author
picks a band explicitly.

## Audio

`render-core` does not capture audio. It expects the **Realtime Audio
Feature Server** (separate Python process, see
`Realtime_PyAudio_FFT/README.md`) to be running and emitting OSC to
`127.0.0.1:9000`. The render thread reads features from a lock-free atomic
snapshot updated by a dedicated UDP recv thread.

Quick start:

```bash
# Terminal 1
audio-server

# Terminal 2
cargo run -- --scene examples/phase3_smoke.scene.json --windowed
```

Drivers `audio.band("low|mid|high")` and `audio.onset(band, decay)` map
onto the server's `/audio/lmh` and `/audio/onset/{low,mid,high}` streams.
The L/M/H values arrive pre-scaled into `~[0, 1]` by the server's
autoscaler; the engine just stores and forwards. Onset arrivals stamp an
engine-local timestamp and `audio.onset` decays `exp(-dt/τ)` on read.

The engine auto-detects the server: the UDP socket is bound at startup
regardless of whether the server is running, and packets flow as soon as
the server starts. Stopping the server freezes the last values;
restarting it resumes mid-flight. Status transitions log one line each
(`OSC: connected (packets arriving)` / `OSC: stale (no packets for >2s)`).

Override the bind address with `--osc-addr 0.0.0.0:9000` (listen on all
interfaces — useful if the audio server runs on another machine), or skip
ingest entirely with `--no-osc` for headless tests.

DSP knobs (gates, compression, onset sensitivity, BPM smoothing) all live
in the audio server's own browser UI — `render-core` is a passive sink.

## What's here vs deferred

In:

- Layer pack loader → single `Texture2DArray<R8>` (256-slice hard cap, D4).
- Strict `scene.json` parser with stable binding ids for hot-reload (§4.2).
- Driver bus: clock, audio (OSC ingest from external feature server), ui-slider stub.
- Effect catalog: `tint`, `hueCycle`, `flash`, `wobble`.
- Inline WGSL effects in `scene.json` (D15).
- Project-local WGSL effects under `effects/<name>/{shader.wgsl, descriptor.json}` (D15).
- `naga`-validated pipeline compile with swap-on-success — a bad save keeps
  the previous good pipeline rendering (§3.6).
- File watcher covers scene.json **and** effects directory.
- Per-layer composite + final homography pass (identity by default).

Out (Phase 4+):

- Slow-path FBO routing for `layerRef` consumers (D5).
- Tauri shell + webview UI (Phase 4).
- Video paths: HAP, hardware-decoded H.264/HEVC (Phase 5).
- JSON-RPC WebSocket / MCP wrapper (Phase 7).

## Example

`examples/phase3_smoke.scene.json` targets the layer pack produced by
`python test.py layerpack` (`test_results/layerpack/pack/`). It exercises
every built-in effect, an inline WGSL effect, drivers (clock + audio), and
the project-local `effects/drift/` effect.

```bash
# Build the pack first:
python test.py layerpack

# Start the audio feature server (separate repo) so OSC features flow:
audio-server

# Then run:
cd render-core
cargo run -- --scene examples/phase3_smoke.scene.json --windowed
# (or --no-osc to skip OSC ingest entirely)
```

Edit any of:

- `examples/phase3_smoke.scene.json` — tweak params, change bindings, swap effects.
- `examples/effects/drift/shader.wgsl` — edit the user shader.
- `examples/effects/drift/descriptor.json` — change the input slot layout.

The projector window updates within one frame budget on save.
