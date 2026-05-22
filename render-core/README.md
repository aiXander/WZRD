# render-core

WZRD's realtime additive projection-mapping engine.

Phase 2 deliverable: standalone Rust + wgpu binary that loads a layer pack
(D3) and a `scene.json` (D13), composites masked layers in z-order, and
applies a final homography pass (D9). One built-in effect — `tint` — drives
the end-to-end pipeline. **No Tauri, no webview, no TypeScript.** Edit
`scene.json` in any editor, save, projector updates within one frame budget.

See `../render_engine_architecture.md` for the full design.

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
```

The layer-pack path is read from `scene.json`'s `pack` field (resolved
relative to the scene file). Override with `--pack`.

## Headless agent loop

```bash
# 1) Build a layer pack from masks + a darkened surface.
python -m wzrd.layerpack masks/ --surface surface.png --tags tags.json \
    -o packs/tree-2026-05-01/

# 2) Author a scene.json that targets it.
cat > scene.json <<'JSON'
{
  "version": 1,
  "pack": "packs/tree-2026-05-01",
  "transport": { "bpm": 120 },
  "bindings": [
    { "id": "trunk", "select": { "id": "trunk" },
      "effect": "tint", "params": { "color": "#3a2818" } },
    { "id": "leaves", "select": { "tag": "leaves" },
      "effect": "tint", "params": { "color": "#0aff44" } }
  ]
}
JSON

# 3) Run.
render-core --scene scene.json --windowed
# Edit scene.json in any editor; the projector updates on save.
```

## What's here vs deferred

In:

- Layer pack loader → single `Texture2DArray<R8>` (256-slice hard cap, D4).
- Strict `scene.json` parser with stable binding ids for hot-reload (§4.2).
- Per-layer composite + final homography pass (identity by default).
- `tint` built-in effect (color × mask).
- File-watched scene hot-reload with swap-on-success (a broken save keeps
  the previous good plan rendering, §3.6).

Out (Phase 3+):

- Effect catalog (`hueCycle`, `flash`, `floodFill`, …) and project-local
  `effects/*.wgsl` + inline-WGSL discovery (D15, §3.6).
- Drivers — clock / audio / midi / osc / ui (§3.7).
- Slow-path FBO routing for `layerRef` consumers (D5).
- JSON-RPC WebSocket and the Tauri shell (Phases 4 / 7).
- Video paths: HAP, hardware-decoded H.264/HEVC (Phase 5).

## Example

`examples/tint_smoke.scene.json` targets the layer pack produced by
`python test.py layerpack` (`test_results/layerpack/pack/`). Run the
Python test first to produce the pack, then:

```bash
render-core \
    --scene examples/tint_smoke.scene.json \
    --windowed
```
