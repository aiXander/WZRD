# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WZRD is a Python toolkit for additive projection mapping. It extracts animated elements from static backgrounds so only changing regions get projected — unchanged areas stay invisible.

## Setup & Commands

```bash
# Install (editable, with all optional deps)
pip install -e ".[islands,mcp]"

# Run any module standalone
python -m wzrd.detect photo.jpg -o cropped.png
python -m wzrd.darken surface.jpg -o dark.png
python -m wzrd.align source.png target.png -o aligned.png
python -m wzrd.subtract_frame generated.png background.png -o creature.png
python -m wzrd.subtract_video animation.mp4 background.png -o output.mp4
python -m wzrd.islands input.png -o regions/
python -m wzrd.reproject regions_dir/ -o output.mp4

# Run smoke tests (all or specific modules)
python test.py --all
python test.py subtract_frame darken

# Run MCP server locally
python -m wzrd_mcp            # default: 0.0.0.0:8787
python -m wzrd_mcp --no-debug # disable verbose tool logging
```

No linting, formatting, or CI pipelines are configured. No pytest — tests use a custom `test.py` with assertions on file existence and return types. Test assets live in `test_imgs/`, outputs go to `test_results/`.

## Architecture

**Two packages** defined in `pyproject.toml`:
- `wzrd/` — Core image/video processing library
- `wzrd_mcp/` — FastMCP server layer wrapping wzrd functions as tools

### wzrd/ package

**Import strategy** (`__init__.py`): Eagerly imports `subtract_frame` and `utils`. All other modules (`darken`, `detect`, `align`, `islands`, `subtract_video`, `reproject`, `prepare_surface`) are lazy-imported via `__getattr__` to avoid loading heavy deps (scikit-learn, etc.) when not needed.

**No centralized config.** Each module defines its own `DEFAULT_*` constants. Do not introduce a config system.

**Key modules and their relationships:**
- `utils.py` — Shared primitives: aspect ratio handling, `extract_creature()`, image I/O helpers
- `subtract_frame.py` — Single-frame background subtraction; calls `utils.extract_creature()`
- `subtract_video.py` — Video pipeline using the same algorithm; expensive mask ops run in parallel via `ThreadPoolExecutor`, creature extraction stays sequential
- `darken.py` — Gradient-weighted histogram equalization in CIELAB space
- `detect.py` — Projection surface detection (color proximity + Otsu + convex hull)
- `align.py` — Feature matching (SIFT/AKAZE + RANSAC) with template matching fallback, ECC sub-pixel refinement
- `islands.py` — KMeans color clustering + connected components (requires scikit-learn)
- `reproject.py` — Layer compositing for island videos
- `prepare_surface.py` — Full surface prep pipeline (detect → align → darken). Returns dict with `'image'` (PIL Image) and optional `'video'` (alignment aid)

### wzrd_mcp/ package

MCP server that wraps wzrd functions as tools.

**Key files:**
- `server.py` — FastMCP instance, tool registry, config loading
- `tools.py` — Tool wrappers for core wzrd functions (subtract, detect, align, darken, prepare_surface, islands, reproject, texture_flow)
- `fal_tools.py` — FAL API integrations (kling_v3_image_to_video, nano_banana_pro)
- `file_io.py` — Input resolution (URL → S3 download, base64 decode, local path) and S3 upload with content-hash naming
- `_log.py` — Terminal logging with colors/timestamps; `@logged_tool` decorator auto-logs tool invocations
- `auth.py` — Bearer token middleware (`WZRD_API_KEY`), optional for local dev
- `modal_app.py` — Modal cloud deployment
- `tools_config.json` — Per-tool activation toggles (tools can be disabled here)

**Tool registration pattern:** Tools use `@mcp.tool()` for MCP registration stacked with `@logged_tool` for automatic parameter logging.

**Flow:** URL/base64 inputs → download to tmp → call wzrd function → upload to S3

**Environment variables (`.env` loaded automatically):**
- `WZRD_API_KEY` — Bearer auth token (optional locally, required on Modal)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BUCKET_NAME` — S3 uploads (not needed for local-only dev)
- `CLOUDFRONT_URL` — Optional CDN root for public URLs
- `FAL_KEY` — Required for FAL tools (kling, nano_banana_pro)
- `MODAL_APP_NAME` — TextureFlow Modal app routing (default: "comfyui-wzrd-STAGE")

## Extras

- **opencv-python pinned** to <4.11.0 (missing macOS wheels above that).
- `deprecated/` folder contains old implementations — not part of the package.
