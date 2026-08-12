# render-core

WZRD's realtime additive projection-mapping engine.

A playable, audio-reactive engine that runs with or without a UI process.
Authors and agents express creativity by writing **WGSL** (inline in
`scene.json` or as project-local files) — the engine hot-reloads on save
(D15). Drivers wire clock + audio (OSC ingest from the external Realtime
Audio Feature Server) straight into effect params.

Two ways to run it: the **standalone binary** from this crate (headless —
one projector window, file-watcher hot-reload; everything below), or the
**Tauri shell** at `../wzrd-app/`, which links this crate as a library and
hosts it in-process. The shell needs no build from here — see
`../README.md`.

See `../docs/reference/render-engine.md` for the full design (roadmap in
`../docs/TODO/render-engine-roadmap.md`).

## Build

Requires Rust 1.85+ (uses the wgpu 22 / winit 0.30 ecosystem).

```bash
cd render-core
cargo build --release
```

> **There is no** `render-core` **on your PATH.** `cargo build` only writes the
> binary into `render-core/target/{debug,release}/`. Launch it as
> `cargo run --` or `./target/release/render-core` — bare `render-core`
> gives you `command not found`, and that is expected.



## Run

One-time, from the **repo root** — the example scenes point at the pack
this builds (`test_results/layerpack/pack/`), and without it the engine
exits on a missing pack:

```bash
uv run python test.py layerpack
```

Then, from `**render-core/**` (all paths below are relative to this
directory):

```bash
# Main test launch — windowed, no audio server needed:
cargo run -- --scene examples/phase3_smoke.scene.json --windowed --no-osc

# Same via the built binary (after `cargo build --release`):
./target/release/render-core --scene examples/phase3_smoke.scene.json --windowed --no-osc

# Borderless fullscreen on the primary monitor (drop --windowed):
cargo run -- --scene examples/phase3_smoke.scene.json

# Pick a specific display:
cargo run -- --scene examples/phase3_smoke.scene.json --display 1

# Release build for an actual show — debug builds drop frames:
cargo run --release -- --scene examples/phase3_smoke.scene.json --display 1

# Bind OSC on all interfaces (audio server on another machine):
cargo run -- --scene examples/phase3_smoke.scene.json --windowed --osc-addr 0.0.0.0:9000

# Expose the JSON-RPC WebSocket so external §3.11 clients (the MCP authoring
# tools, a remote operator) can drive scene / effect / telemetry.
# (The Tauri shell does NOT need this — it embeds the engine in-process.)
# NOTE: --ws-addr also switches the engine into §5.6 two-deck mode — all
# authoring lands on the design leg and goes live via `promote`. Headless
# (no --ws-addr) stays single-leg: the watcher hot-reloads live directly.
cargo run -- --scene examples/phase3_smoke.scene.json --windowed --ws-addr 127.0.0.1:9123
```

`--scene` is the only required flag. `cargo run -- --help` prints the full
list; the rest are `--pack`, `--effects`, `--display`, `--windowed`,
`--no-osc`, `--osc-addr`, `--ws-addr`, `--frame-cap-hz` (default 240, `0`
disables the cap).

The layer-pack path is read from `scene.json`'s `pack` field (resolved
relative to the scene file). Override with `--pack`.

Project-local effects (D15) are picked up automatically from
`<scene_dir>/effects/`. Override with `--effects path/to/effects/`.

## Effects (Phase 3 baseline)

Built-ins shipped as reference implementations — **not the boundary** of
what's expressible:


| name       | inputs                         | notes                                                 |
| ---------- | ------------------------------ | ----------------------------------------------------- |
| `tint`     | `color`                        | color × mask (debug / baseline)                       |
| `hueCycle` | `phase`, `color0..color3`      | 4-stop palette cycled by `phase` (often `clock.bars`) |
| `flash`    | `envelope`, `base`, `color`    | additive pulse, typically `envelope = audio.onset`    |
| `wobble`   | `amp`, `freq`, `time`, `color` | UV displacement — edge undulates                      |


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
# Terminal 1 — the audio server (separate repo)
cd ~/Documents/GitHub/Realtime_PyAudio_FFT && uv run audio-server --open

# Terminal 2 — the engine, from render-core/
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
- Driver bus: clock, audio (OSC ingest from external feature server),
`ui.slider` (live values written by `param.set`, persisted in `session.json`).
- Effect catalog: `tint`, `hueCycle`, `flash`, `wobble`.
- Inline WGSL effects in `scene.json` (D15).
- Project-local WGSL effects under `effects/<name>/{shader.wgsl, descriptor.json}` (D15).
- `naga`-validated pipeline compile with swap-on-success — a bad save keeps
the previous good pipeline rendering (§3.6).
- File watcher covers scene.json **and** effects directory.
- Per-layer composite + final pass: §2.8 alignment warp (n-point, baked to
an offset LUT; identity until `alignment.json` says otherwise) + output
masters + the §5.6 promote crossfade.
- JSON-RPC WebSocket control surface (`--ws-addr`), which also enables the
§5.6 design/live two-deck. Consumed by the Tauri shell (in-process, no
socket) and the `wzrd_mcp` authoring tools.

Still out:

- Slow-path FBO routing for `layerRef` consumers (D5) — parsed, not executed.
- `post` bindings — parsed, not executed.
- Video paths: HAP, hardware-decoded H.264/HEVC (roadmap §5.9).



## Example

Two scenes ship here, both targeting the layer pack built by
`uv run python test.py layerpack` (`test_results/layerpack/pack/`). Its
five regions are named after the test photo (a Moroccan kasbah) —
`background`, `main_facade`, `left_tower`, `right_wing`, `base_wall`, with
a `towers` group and `facade` / `tower` / `wall` tags — because those ids
are what a scene selects on and what the operator reads in the inspector.
The names are authored in `test.py::test_layerpack`'s tags file; without a
tags entry a region falls back to its islands mask stem
(`region_mask_003_color_fefd00`), which is unusable in a binding list:

- `examples/phase3_smoke.scene.json` — the full exercise: built-in effects,
an inline WGSL effect, clock + audio drivers, and the project-local
effects under `examples/effects/` (`bloom`, `drift`, `flow_noise`,
`vbars`).
- `examples/tint_smoke.scene.json` — minimal `tint`-only scene; the
fastest way to confirm the pack loads and the compositor is alive.

```bash
# 1. Build the pack first (from the repo root, one-time):
uv run python test.py layerpack

# 2. Optional — audio features over OSC (separate repo):
cd ~/Documents/GitHub/Realtime_PyAudio_FFT && uv run audio-server --open

# 3. Run, from render-core/:
cargo run -- --scene examples/phase3_smoke.scene.json --windowed
# drop step 2 and add --no-osc to skip OSC ingest entirely
```

Edit any of these and the projector window updates within one frame budget
on save:

- `examples/phase3_smoke.scene.json` — tweak params, change bindings, swap effects.
- `examples/effects/<name>/shader.wgsl` — edit the user shader.
- `examples/effects/<name>/descriptor.json` — change the input slot layout.

## Aligning the output without a UI (§2.8)

`alignment.json` next to the scene is engine-written and applied whether or
not anything is attached, so a saved alignment survives a headless run. To
drive it live you need a control surface (`--ws-addr`, which the Tauri shell
also binds):

```bash
cargo run -- --scene examples/phase3_smoke.scene.json --windowed --no-osc \
    --ws-addr 127.0.0.1:9123

# in another shell — corner sweep, live handle demo, test patterns:
uv run --with websockets tools/align_drag.py --demo sweep
uv run --with websockets tools/align_drag.py --pattern grid
uv run --with websockets tools/align_drag.py --get
```

`tools/align_drag.py` is deliberately kept rather than thrown away: it is the
proof that the alignment surface is UI-independent, and the skeleton the
camera-driven auto-align (capture → detect → solve → `alignment.set`) grows
from. `--verify-isolation <alignment.json>` asserts a `scene.load` + `pull`
leaves the file byte-identical.

