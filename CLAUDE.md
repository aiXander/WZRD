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

# Run smoke tests
python test.py

# Run MCP server locally
python -m wzrd_mcp  # default: 0.0.0.0:8787
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

### wzrd_mcp/ package

MCP server that wraps wzrd functions as tools. Key files:
- `server.py` — FastMCP instance, tool registry, config loading
- `tools.py` — Tool wrappers for each wzrd function
- `file_io.py` — Input resolution (URL → S3 download, base64 decode, local path) and S3 upload
- `modal_app.py` — Modal cloud deployment
- `tools_config.json` — Per-tool activation toggles

Flow: URL/base64 inputs → download to tmp → call wzrd function → upload to S3

## Extras

- **opencv-python pinned** to <4.11.0 (missing macOS wheels above that).
- `deprecated/` folder contains old implementations — not part of the package.