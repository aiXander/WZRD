# WZRD - Projection Mapping Toolkit

A Python toolkit for additive projection mapping. Extracts animated elements from static backgrounds so only the changing regions get projected — unchanged areas stay invisible, making characters appear to inhabit real surfaces.

WZRD has **two halves**:

1. **Offline Python pipeline** (`wzrd/`) — prep tools that turn a photo of your surface + generated animations into a **layer pack** (`pack.json` + masks + surface). This is everything in the [Modules](#modules) / [Python API](#python-api) sections below.
2. **Realtime render engine + GUI** (`render-core/` Rust crate, `wzrd-app/` Tauri shell) — plays that layer pack live on a projector, audio-reactive, with WGSL effects that hot-reload on save. This is the [Realtime Render Engine & GUI](#realtime-render-engine--gui) section — jump there if you just want to *run the visuals*.

---

## Realtime Render Engine & GUI

**Quick start (copy-paste):**

```bash
# Full app — UI + engine + native preview, ONE command (single process since 2026-07-12):
cd wzrd-app && WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev

# Engine only, headless (agents / live shows, no UI):
cd render-core && cargo run -- --scene examples/phase3_smoke.scene.json --windowed --no-osc
```

> **The one gotcha:** there is no `render-core` command on your PATH. `cargo build` only writes the binary into `render-core/target/`. Always launch it with `cargo run --` or the explicit binary path `./target/release/render-core`. Typing bare `render-core` gives `command not found` — that's expected.

There are **two ways to run** the engine, plus an optional audio server that feeds both:


|                     | What runs                                                        | When to use                                                                |
| ------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Headless engine** | `render-core` alone, a single projector window                   | Live shows, agent deployment, quick iteration on a scene/shader            |
| **GUI shell**       | Tauri app (`wzrd-app`) running `render-core` in-process — one app, engine window + native preview included | Authoring: Monaco editor, mask overlays, binding inspector, live telemetry |


### Prerequisites (one-time)

```bash
# 1. Rust 1.85+ toolchain (wgpu 22 / winit 0.30). Install via https://rustup.rs
# 2. Build the example layer pack the sample scene points at:
python test.py layerpack        # writes test_results/layerpack/pack/
```

### Option A — Headless engine (fastest path to pixels)

```bash
cd render-core

# Debug build is fine for iterating; --release is smoother for live shows.
cargo run -- --scene examples/phase3_smoke.scene.json --windowed --no-osc
```

- `--windowed` — run in a window instead of borderless-fullscreen (drop it to go fullscreen on the primary display; add `--display 1` to pick a monitor).
- `--no-osc` — **skip the audio server**. Clocks still tick and all `clock.`* effects animate; `audio.*` drivers just return 0 (so the audio-reactive `flash`/`drift` layers sit at their defaults). Start the audio server (below) and drop `--no-osc` to make them react.
- The scene's `pack` field (resolved relative to the scene file) points at the layer pack. Override with `--pack path/to/pack/`.
- Any WGSL under `render-core/examples/effects/<name>/` and inline WGSL in the scene hot-reload the instant you save.

Prefer the prebuilt binary over `cargo run` once compiled:

```bash
./target/release/render-core --scene examples/phase3_smoke.scene.json --windowed --no-osc
```

### Option B — GUI control shell (authoring)

The Tauri app runs the engine **in-process** (single process, since 2026-07-12), so you get the projector window **plus** the Prepare/Perform/Debug UI plus a native lossless preview on the Perform route.

```bash
cd wzrd-app
pnpm install        # one-time
WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev

# with audio server auto-launch:
WZRD_AUDIO=1 WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev
```

- No separate engine build step — `render-core` compiles in as a library dependency.
- Routes switch with `⌘1` (Prepare) / `⌘2` (Perform) / `⌘3` (Debug).
- `WZRD_DISPLAY=<idx>` puts the engine output window borderless-fullscreen on that monitor (default: a regular window at pack resolution).
- `pnpm tauri dev` runs the engine **with OSC enabled**, so `audio.`* layers stay at defaults until the audio server is up. See below.

### Optional — Realtime Audio Feature Server (makes `audio.*` drivers react)

`render-core` does **not** capture audio itself. Audio features (`audio.band`, `audio.onset`) arrive over OSC from a separate project, `[Realtime_PyAudio_FFT](https://github.com/xandersteenbrugge/Realtime_PyAudio_FFT)`, listening on `127.0.0.1:9000` by default.

```bash
# In its own terminal, from the audio server repo:
cd ~/Documents/GitHub/Realtime_PyAudio_FFT
uv run audio-server --open      # starts the server + opens its browser UI
```

Start it before (or during — it auto-connects mid-flight) either run option, and drop `--no-osc`. The engine binds the UDP socket at startup whether or not the server is running, so packets flow the moment it comes up and freeze if it stops.

**Full live-show stack, three terminals:**

```bash
# Terminal 1 — audio
cd ~/Documents/GitHub/Realtime_PyAudio_FFT && uv run audio-server --open

# Terminal 2 — engine (fullscreen on projector = display 1)
cd render-core && cargo run --release -- --scene examples/phase3_smoke.scene.json --display 1

# ...or Terminal 2 alternative — GUI shell instead of headless engine
cd wzrd-app && WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev
```

See `[render-core/README.md](render-core/README.md)` for the effect/driver catalog and WGSL prelude, and `[wzrd-app/README.md](wzrd-app/README.md)` for the shell's routes and telemetry channels.

---

## Installation

```bash
pip install -e .

# With island extraction support (requires scikit-learn for KMeans)
pip install -e ".[islands]"

# From GitHub
pip install "wzrd @ git+https://github.com/xandersteenbrugge/WZRD.git"
```

**Requirements:** Python 3.10+, FFmpeg (for video processing)

## Modules

Every module runs standalone via `python -m wzrd.<module>`. All processing parameters have sensible defaults.

### detect — Projection Surface Detection

Automatically finds the projection area in a photo of the lit surface. Uses color proximity mapping, Otsu thresholding, and convex hull fitting.

```bash
python -m wzrd.detect photo.jpg -o cropped.png --margin 0.02 --aspect-ratio 16:9
```

### darken — Gradient-Weighted Darkening

Darkens well-lit surface photos into faint guidance images for AI generation. Uses gradient-weighted histogram equalization in CIELAB space to preserve texture detail while reducing overall brightness.

```bash
python -m wzrd.darken surface.jpg -o dark.png --max-brightness 0.25 --aspect 16:9
```

### align — Image Alignment

Aligns a source image to a target via feature matching (SIFT/AKAZE + RANSAC) with template matching fallback and ECC sub-pixel refinement.

```bash
python -m wzrd.align source.png target.png -o aligned.png
```

### subtract_frame — Single-Frame Background Subtraction

Isolates the "creature" (changed regions) from a generated frame against its background. Pipeline: color correction → LAB difference → soft ramp threshold → morphological cleanup → guided filter feathering → gamma-corrected extraction.

```bash
python -m wzrd.subtract_frame generated.png background.png -o creature.png --preview
```

### subtract_video — Video Background Subtraction

Processes entire videos frame-by-frame with multi-resolution acceleration and temporal EMA smoothing. Color correction runs at 1/4 res, masks at 1/2 res, guided filter at full res.

```bash
python -m wzrd.subtract_video animation.mp4 background.png -o output.mp4 --temporal-smoothing 0.3
```

### islands — Color Region Extraction

Segments images into distinct color regions via K-means clustering, extracts connected components, and saves per-region crops with JSON metadata for reconstruction.

```bash
python -m wzrd.islands input.png -o regions/ --max-colors 6 --surface surface.png
```

### reproject — Layer Compositing

Reprojects cropped island videos back onto a full-size canvas at their original positions, for layer-based compositing in tools like Resolume.

```bash
python -m wzrd.reproject regions_dir/ -o output.mp4 --aspect 16:9
```

## Python API

```python
import wzrd

# Background subtraction (single frame)
creature, mask, info = wzrd.subtract_background(generated, background,
    threshold=10, ramp=20, gamma=0.85, diff_mode='lab')

# Background subtraction (video)
wzrd.subtract_background_video("video.mp4", "background.png",
    output_path="output.mp4", temporal_smoothing=0.3)

# Darken image
from wzrd import darken_image
darkened = darken_image(image, max_brightness=0.25)

# Detect projection surface
from wzrd import detect_projection_area
detect_projection_area("photo.jpg", margin=0.02, target_aspect_ratio=16/9)

# Align images
from wzrd import align_images
warped, result = align_images(source, target)

# Extract color regions
from wzrd import extract_color_regions
extract_color_regions(image, output_dir="regions/", max_colors=6)

# Reproject layers
from wzrd import reproject_videos_batch
reproject_videos_batch(video_json_pairs, canvas_width=1920, canvas_height=1080)
```

## MCP Server

Deploy mcp server to modal:
`modal deploy wzrd_mcp/modal_app.py`

WZRD includes an MCP (Model Context Protocol) server that exposes all tools to AI agents.

### Running the server

```bash
pip install -e ".[mcp]"
python -m wzrd_mcp                    # default: 0.0.0.0:8787
python -m wzrd_mcp --port 8000        # custom port
```

The server uses Streamable HTTP transport at `http://localhost:8787/mcp`. It won't respond to regular browser requests (you'll get a 406 — that's normal).

### Inspecting & testing tools

Use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) to browse and invoke tools interactively:

```bash
npx @modelcontextprotocol/inspector
```

This opens a UI at **[http://localhost:5173](http://localhost:5173)**. To connect:

1. Set transport type to **Streamable HTTP**
2. Enter URL: `http://localhost:8787/mcp`
3. Click **Connect**

You'll see all registered tools with their schemas and can fire test calls.

### Connecting from Claude Code

Add to your MCP config (e.g. `.claude/settings.json`):

```json
{
  "mcpServers": {
    "wzrd": {
      "type": "streamable-http",
      "url": "http://localhost:8787/mcp"
    }
  }
}
```

## Dependencies

- **Pillow**, **NumPy**, **OpenCV** (core)
- **FFmpeg** (system install, for video I/O)
- **scikit-learn** (optional, for `islands` color clustering)

## Tips

- **Threshold tuning:** Start low (5–10) and increase if background bleeds through
- **Diff mode:** `lab` is most perceptually accurate; `luminance` is faster; `rgb` is simplest
- **Temporal smoothing:** 0.3 gives moderate stability; 0 disables for per-frame independence
- **Gamma < 1.0** brightens extracted creatures (default 0.85), useful for projection
- **Feathering:** Higher radius gives softer edges but costs performance in video mode

