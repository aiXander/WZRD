# WZRD — Additive Projection Mapping

Only the *changing* regions get projected — unchanged areas stay dark and merge with the physical surface, so animated elements appear to inhabit real objects.

Three parts:

1. **Offline Python toolkit** (`wzrd/`) — turns a photo of your surface + generated animations into a **layer pack** (`pack.json` + masks + surface). → [CLI modules](#offline-toolkit--cli-modules)
2. **Realtime engine** (`render-core/`, Rust + wgpu) — plays that pack on a projector, audio-reactive, with WGSL effects that hot-reload on save.
3. **Control shell** (`wzrd-app/`, Tauri) — the human authoring UI; runs the engine **in-process** (one app, one process).

---

# Quick start

## 0. Install (one-time)

```bash
uv sync --extra islands --extra mcp --extra engine   # everything
# minimal alternatives: uv sync                (base toolkit)
#                       uv sync --extra islands (color-region extraction)
# plain pip fallback:   pip install -e ".[islands,mcp,engine]"

uv run python test.py layerpack   # builds test_results/layerpack/pack/ — the pack the sample scene points at
```

Also needed: **Rust 1.85+** ([rustup.rs](https://rustup.rs)) for the engine, **pnpm** for the shell, **FFmpeg** for video I/O. Python 3.10+.

## 1. Run the visuals

```bash
# GUI shell — UI + engine window + native preview, one process, one command:
cd wzrd-app && pnpm install     # one-time
WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev

# Headless engine — no UI (live shows, agent deployment, quick shader iteration):
cd render-core && cargo run -- --scene examples/phase3_smoke.scene.json --windowed --no-osc
```

| | GUI shell (`wzrd-app`) | Headless engine (`render-core`) |
|---|---|---|
| Gives you | Prepare/Perform/Debug routes, Monaco editor, mask overlays, binding inspector, deck (LIVE⇄DESIGN), telemetry | A single projector window, file-watcher hot-reload |
| Control surface | always binds `ws://127.0.0.1:9123` | only with `--ws-addr 127.0.0.1:9123` |
| Engine build | none — `render-core` links in as a library | `cargo run` / `cargo build --release` |

**`render-core` flags** (`cargo run -- --help` for the full list):

- `--windowed` — window instead of borderless fullscreen; `--display 1` picks a monitor.
- `--no-osc` — skip the audio server: clocks still tick, `audio.*` drivers return 0.
- `--pack path/to/pack/` — override the scene's `pack` field (otherwise resolved relative to the scene file).
- `--effects dir/`, `--osc-addr`, `--ws-addr`, `--frame-cap-hz`.

**Shell env vars** (it configures the engine itself, not via those flags): `WZRD_SCENE` (or `--scene`), `WZRD_DISPLAY=<idx>` for borderless-fullscreen on that monitor, `WZRD_AUDIO=1` to auto-start the audio server. The shell always runs the engine **with** OSC enabled. Routes switch with `⌘1` Prepare / `⌘2` Align / `⌘3` Perform / `⌘4` Debug.

Either way, WGSL under `<scene_dir>/effects/<name>/` and inline WGSL in the scene hot-reload the instant you save.

> **The one gotcha:** there is no `render-core` on your PATH. `cargo build` only writes the binary into `render-core/target/`, so launch via `cargo run --` or `./target/release/render-core`. Bare `render-core` → `command not found`, expected.

## 2. Audio reactivity (optional)

The engine does **not** capture audio. `audio.band` / `audio.onset` arrive over OSC on `127.0.0.1:9000` from a separate repo, [Realtime_PyAudio_FFT](https://github.com/xandersteenbrugge/Realtime_PyAudio_FFT):

```bash
cd ~/Documents/GitHub/Realtime_PyAudio_FFT && uv run audio-server --open
```

Start it any time — it auto-connects mid-flight — and drop `--no-osc`. The shell can spawn it for you with `WZRD_AUDIO=1` (killed on shell close; override its location with `WZRD_AUDIO_DIR`).

## 3. AI scene authoring (Claude Desktop or Claude Code + running engine)

A **locally-run** `wzrd_mcp` exposes the engine's authoring slice so an AI session can design scenes and write WGSL against the live engine. The agent only ever touches the **design** leg — going live stays a human act.

Works from either client. Pick one:

**Claude Desktop** ([setup](#connecting-claude-desktop)) — one-time config, then the only thing you launch per session is the engine:

```bash
# one-time: add the config block from "Connecting Claude Desktop" below, then ⌘Q + reopen
# per session — just start the engine; Desktop spawns the MCP server itself:
cd wzrd-app && WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev
```

**Claude Code** — the MCP server runs as its own HTTP process:

```bash
# a) engine up with a control surface — either of:
cd wzrd-app && WZRD_SCENE=../render-core/examples/phase3_smoke.scene.json pnpm tauri dev
cd render-core && cargo run -- --scene examples/phase3_smoke.scene.json --windowed --ws-addr 127.0.0.1:9123

# b) MCP server (its own terminal)
uv run python -m wzrd_mcp

# c) register once, then open a fresh Claude Code session in this repo
claude mcp add --transport http wzrd http://localhost:8787/mcp
```

Restart the engine as often as you like in either case — the tools re-dial on the next call.

**The loop** — you type intent, the agent authors, you referee:

| Say | Tool | What comes back |
|---|---|---|
| "what's on the surface right now?" | `get_scene_context` | digest: layers with labels/groups, bindings with resolved targets, effects, live driver values, status header + recent changes |
| "call the big center region 'trunk'" | `set_labels` / `set_groups` | writes the `identity.json` sidecar — "the trunk" resolves from then on |
| "make the trunk pulse purple on the bass" | `upsert_binding` / `upsert_effect` | naga diagnostics or a probe verdict (green/yellow/red) **plus a thumbnail of the design composite**, so the agent self-corrects without you refereeing |
| "show me what it looks like" | `get_preview` | one design-leg JPEG |

**Your seat (the Tauri UI):**

- Keep the DeckBar preview toggle on **DESIGN** — that's where the agent's work shows. The projector (LIVE) is untouched.
- Agent edits reverse-sync into the UI automatically (bindings, labels, groups) — no reload.
- An amber **ADOPT AGENT SCENE · rN** button appears in the TopBar. Clicking it is the *only* thing that writes agent work into `scene.json`. Until then the durable copy lives at `<scene_dir>/.wzrd/scene_agent_latest.json`, rewritten on every agent apply.
- When the design preview looks right, hit **PROMOTE** in the DeckBar. There is no agent promote tool, ever.

Notes: tools connect to `ws://127.0.0.1:9123` (`WZRD_ENGINE_WS` overrides). If the engine is down they return one prescriptive `engine unreachable since <time>` line instead of hanging — start it and re-ask. An effect is only probed (verdict + thumbnail) once the scene actually **binds** it. Full contract: [docs/reference/render-engine.md §2.7](docs/reference/render-engine.md).

## Full live-show stack (three terminals)

```bash
# 1 — audio
cd ~/Documents/GitHub/Realtime_PyAudio_FFT && uv run audio-server --open
# 2 — engine, fullscreen on the projector (or the GUI shell instead)
cd render-core && cargo run --release -- --scene examples/phase3_smoke.scene.json --display 1
# 3 — MCP server, only if you're co-authoring from Claude Code (Desktop spawns its own)
uv run python -m wzrd_mcp
```

Deeper reference: [`docs/reference/render-engine.md`](docs/reference/render-engine.md) (contracts, invariants — read before touching the engine), [`render-core/README.md`](render-core/README.md) (effect/driver catalog, WGSL prelude), [`wzrd-app/README.md`](wzrd-app/README.md) (routes, telemetry channels).

---

# Offline toolkit — CLI modules

Every module runs standalone via `python -m wzrd.<module>`; all parameters have sensible defaults.

```bash
# Find the projection area in a photo of the lit surface (color proximity + Otsu + convex hull)
python -m wzrd.detect photo.jpg -o cropped.png --margin 0.02 --aspect-ratio 16:9

# Darken a lit surface photo into a faint guidance image for AI generation (CIELAB, gradient-weighted)
python -m wzrd.darken surface.jpg -o dark.png --max-brightness 0.25 --aspect 16:9

# Align a source image to a target (SIFT/AKAZE + RANSAC, template-match fallback, ECC refinement)
python -m wzrd.align source.png target.png -o aligned.png

# Isolate the "creature" (changed regions) from a generated frame against its background
python -m wzrd.subtract_frame generated.png background.png -o creature.png --preview

# Same, over a whole video (expensive mask ops run in parallel; ffmpeg codec is a flag)
python -m wzrd.subtract_video animation.mp4 background.png -o output.mp4 --threshold 10 --codec libx264

# Segment into color regions via K-means + connected components (needs the `islands` extra)
python -m wzrd.islands input.png -o regions/ --max-colors 6 --surface surface.png

# Reproject cropped island videos back onto a full-size canvas
python -m wzrd.reproject regions_dir/ -o output.mp4 --aspect 16:9

# Author the layer pack the realtime engine consumes
python -m wzrd.layerpack masks_dir/ --surface surface.png --tags tags.json -o pack/
```

Smoke tests: `python test.py --all`, or a subset: `python test.py subtract_frame darken`. Assets in `test_imgs/`, outputs in `test_results/`.

## Python API

```python
import wzrd

creature, mask, info = wzrd.subtract_background(generated, background,
    threshold=10, ramp=20, gamma=0.85)
wzrd.subtract_background_video("video.mp4", "background.png",
    output_path="output.mp4", threshold=10, codec="libx264")

from wzrd import darken_image, detect_projection_area, align_images
from wzrd import extract_color_regions, reproject_videos_batch

darkened = darken_image(image, max_brightness=0.25)
detect_projection_area("photo.jpg", margin=0.02, target_aspect_ratio=16/9)
warped, result = align_images(source, target)
extract_color_regions(image, output_dir="regions/", max_colors=6)
reproject_videos_batch(video_json_pairs, canvas_width=1920, canvas_height=1080)
```

### Tips

- **Threshold:** start low (5–10), raise if background bleeds through; `--ramp` controls how soft that cutoff is.
- **Gamma < 1.0** brightens extracted creatures (default 0.85) — useful for projection.
- **`--blur-radius`:** larger = softer mask edges, costs performance in video mode.
- **Color correction** is on by default (it compensates for the generator shifting the whole frame); `--no-color-correction` if it fights you.

---

# MCP server

Exposes the toolkit (and, locally, the engine authoring slice) to AI agents. Two transports off the same tool registry: **Streamable HTTP** at `http://localhost:8787/mcp` (Claude Code, Modal, the inspector — a plain browser request returns 406, that's normal) and **stdio** for clients that spawn the server themselves (Claude Desktop).

```bash
uv run python -m wzrd_mcp                # HTTP on 0.0.0.0:8787
uv run python -m wzrd_mcp --no-debug     # quieter (no per-tool logging)
uv run python -m wzrd_mcp --port 8000    # custom port
uv run python -m wzrd_mcp --stdio        # stdio; you rarely run this by hand — the client does

modal deploy wzrd_mcp/modal_app.py       # cloud deployment (HTTP only) — PARKED, see below
```

## Everything is local: project folders

Tool outputs never leave this machine. Each job gets a folder under `projects/`
(gitignored, override the root with `WZRD_PROJECTS_ROOT`), and every tool
returns **absolute local paths** — which every tool also accepts as input, so
steps chain path-to-path:

```
projects/<name>/
  scene.json          engine entry point   ← `pack` resolves relative to here
  effects/            project-local WGSL   ← render-core looks here by default
  packs/<pack>/       layer packs (pack.json + masks/ + references/)
  surfaces/           prepared surfaces + alignment aid videos
  generated/          model outputs (FAL, …)
  video/              subtract / reproject / simulate output
  regions/            extract_color_regions runs
  captures/           camera snapshots
```

Call **`set_project("my_gig")`** first in a session to pick the folder (or set
`WZRD_PROJECT`; the fallback is `default`). A project folder *is* a scene
directory, so `build_layerpack` output is loadable by the engine as-is:
`cargo run -- --scene projects/my_gig/scene.json`.

Remote model calls (FAL) download their results into the project. Local paths
handed to a FAL tool are uploaded to *FAL's own* CDN just in time, since its
workers can't read your disk — the only outbound copy left, and an ephemeral
one. `wzrd_mcp/file_io.py`'s S3 upload path is kept but **dormant**: no tool
calls it, `boto3` moved to the optional `s3` extra, and it's there for the day
it gets pointed at a self-owned bucket.

- **Per-tool toggles** live in `wzrd_mcp/tools_config.json` (enabled flag + timeout).
- **Engine tools** (`get_scene_context`, `upsert_binding`, `upsert_effect`, `set_groups`/`set_labels`, `validate_wgsl`, `get_preview`, …) need the `engine` extra → `websockets`. They default **off** in `server.py` and are on in the local `tools_config.json`; the Modal image never installs `websockets`, so the cloud deployment structurally can't carry them. Usage: [§3 above](#3-ai-scene-authoring-claude-desktop-or-claude-code--running-engine).
- **Register in Claude Code:** `claude mcp add --transport http wzrd http://localhost:8787/mcp`.
- **Register in Claude Desktop:** stdio, no running server needed — [see below](#connecting-claude-desktop).
- **Browse/test tools interactively:** `npx @modelcontextprotocol/inspector` → open http://localhost:5173, transport **Streamable HTTP**, URL `http://localhost:8787/mcp`, Connect.

## Connecting Claude Desktop

Desktop talks to **local** servers over **stdio**: it spawns the process itself, on demand. There is no server to start, no port, no second terminal — you hand it a command line once and it takes over from there. Five steps, ~2 minutes.

### 1. Install the deps

```bash
cd /path/to/WZRD
uv sync --extra mcp --extra engine --extra islands
# mcp = required · engine = the live-authoring tools (drop for offline-only) · islands = extract_color_regions
```

### 2. Smoke-test the stdio transport

Before involving Desktop, prove the server speaks the protocol. One line in, one line of JSON out, exits immediately:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}' \
  | uv run python -m wzrd_mcp --stdio
```

Expect `{"jsonrpc":"2.0","id":1,"result":{...,"serverInfo":{"name":"WZRD",...}}}`. If you get a traceback or *any* non-JSON text on stdout, fix that first — Desktop will fail the same way, just silently.

### 3. Generate your config block

Both paths must be **absolute** (step 5 explains why). Run this from the repo root and it prints the block filled in with your real paths:

```bash
printf '{\n  "mcpServers": {\n    "wzrd": {\n      "command": "%s",\n      "args": ["--directory", "%s", "run", "python", "-m", "wzrd_mcp", "--stdio"]\n    }\n  }\n}\n' "$(which uv)" "$(pwd)"
```

```json
{
  "mcpServers": {
    "wzrd": {
      "command": "/Users/<you>/.local/bin/uv",
      "args": ["--directory", "/absolute/path/to/WZRD", "run", "python", "-m", "wzrd_mcp", "--stdio"]
    }
  }
}
```

### 4. Paste it into Desktop's config and restart

Claude menu in the **macOS menu bar** (not the in-window settings) → **Settings → Developer → Edit Config**. That opens `~/Library/Application Support/Claude/claude_desktop_config.json` (Windows: `%APPDATA%\Claude\claude_desktop_config.json`). Merge the block into `mcpServers` — keep any servers already there.

Then **fully quit and reopen Claude Desktop** — ⌘Q, not just closing the window. Config is read once at launch.

### 5. Verify and use it

Click the **/** ("Add files, connectors, and more") indicator by the message box → **Connectors** → `wzrd` should be listed with its tools. Then just ask in plain language:

| Ask | Runs | Needs |
|---|---|---|
| "prep this surface photo for projection: `/abs/path/surface.jpg`" | `prepare_surface` (detect → align → darken) | nothing |
| "segment `/abs/path/frame.png` into color regions" | `extract_color_regions` | `islands` extra |
| "what's on the surface right now?" | `get_scene_context` | engine running ↓ |
| "make the trunk pulse purple on the bass" | `upsert_binding` / `upsert_effect` | engine running ↓ |

Note the sub-steps (`darken_surface`, `align_images`, `detect_projection_surface`, `subtract_background_frame`) are **disabled** in `tools_config.json` and won't appear — `prepare_surface` covers them. Flip `enabled` there if you want them exposed.

The **engine tools** only work while the engine is up with its control surface on `ws://127.0.0.1:9123` (`pnpm tauri dev`, or `cargo run -- --ws-addr 127.0.0.1:9123`) — otherwise they return one `engine unreachable` line. Everything from [§3](#3-ai-scene-authoring-claude-desktop-or-claude-code--running-engine) applies unchanged in Desktop: the agent authors the **design** leg, you PROMOTE.

### "Why does Desktop run the server? I'd rather run it myself and restart it freely."

Reasonable instinct — but for the thing you actually restart, it changes nothing, and for the other thing it doesn't buy what it looks like it buys.

- **Restarting the *engine* is already free.** It's a separate process on `ws://127.0.0.1:9123`; the MCP layer holds a connection but drops it to `None` when it breaks and re-dials on the next call. Kill and relaunch the Tauri shell or `render-core` as often as you like — the next tool call reconnects. Nothing to restart on the MCP side. This is the loop that matters during a session.
- **Restarting the *MCP server* costs a reconnect either way.** Running it standalone over HTTP does *not* let you swap it out underneath a live client: MCP sessions are stateful, so when the server restarts the old session id is gone. Verified here — with a bridge held open, `tools/list` succeeded before the restart and hung with no response after it, even though the bridge process was still alive and the new server was up. You'd be re-connecting Desktop regardless, having gained a port, a bridge process, and a terminal to babysit.

So stdio is the same number of steps minus the moving parts, and you never manage the process: each Desktop launch spawns your current code. To pick up edits to tool code, try toggling `wzrd` off/on under Settings → Connectors first — ⌘Q and reopen is the guaranteed one.

Keep the HTTP server for what it's genuinely better at: **Claude Code** (`claude mcp add`), the MCP **inspector**, and the **Modal** deployment. Both transports serve the identical registry, and running both at once is fine — they're separate processes with separate engine connections.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Server missing or dead after restart | Almost always a **non-absolute `command`**. Desktop launches from Finder and never sources your shell profile, so `"command": "uv"` doesn't resolve. Use `which uv` (typically `~/.local/bin/uv`). Same trap bites `npx` under `nvm`, and pyenv/conda shims. |
| Connects, then drops on first tool call | Something printed to **stdout**. Under stdio, stdout *is* the JSON-RPC stream — tool logging must go to stderr (that's what `_log._emit` is for) and the FastMCP banner is suppressed via `show_banner=False`. |
| Tools appear but can't find files | Desktop's working directory is `/`, not your repo. Pass **absolute** paths or URLs. |
| Wrong deps / import errors | The `--directory` arg is what makes `uv` resolve the repo's venv and load `.env`. Don't drop it. |
| Need the actual error | `tail -n 30 -f ~/Library/Logs/Claude/mcp*.log` — `mcp-server-wzrd.log` is this server's stderr. Or reproduce outside Desktop by running the config's `command` + `args` verbatim in a terminal. |

> **Don't use Settings → Connectors → Add custom connector for this.** That dialog is for *remote* servers and rejects anything not `https` ("URL must start with 'https'"), so a local `http://localhost:8787/mcp` can't go there — it's the right place for a deployed Modal URL instead. If you ever must point Desktop at an HTTP endpoint, use `npx -y mcp-remote <url> --transport http-only` as the `command`.

Environment (`.env` is loaded automatically): `FAL_KEY` for the FAL tools (the only key a local session needs), `WZRD_PROJECT` / `WZRD_PROJECTS_ROOT` to preselect where output lands, `WZRD_API_KEY` (bearer auth — only meaningful for a remote deployment). The `AWS_*` / `CLOUDFRONT_URL` vars are read by nothing on the local path any more; they only wake the dormant S3 code in `file_io.py`. See `.env.example`.
