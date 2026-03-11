"""FastMCP server definition for WZRD."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Tool activation config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "tools_config.json"
_DEFAULT_TOOLS = {
    "subtract_background_frame": True,
    "subtract_background_video": True,
    "detect_projection_surface": True,
    "align_images": True,
    "darken_surface": True,
    "prepare_surface": True,
    "extract_color_regions": True,
    "reproject_video": True,
    "texture_flow": True,
    "kling_v3_image_to_video": True,
    "nano_banana_pro": True,
}


def _load_tool_config() -> dict:
    """Load tool activation config. Falls back to all-enabled defaults."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return {**_DEFAULT_TOOLS, **json.load(f)}
    return dict(_DEFAULT_TOOLS)


TOOL_CONFIG = _load_tool_config()

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "WZRD",
    instructions=(
        "WZRD is a VJ projection mapping toolkit. It provides tools for:\n"
        "- Background subtraction (frame and video) for extracting creatures/subjects\n"
        "- Projection surface detection from photos\n"
        "- Image alignment (feature matching + template matching + ECC)\n"
        "- Surface darkening for additive projection\n"
        "- Color region segmentation (islands)\n"
        "- Video reprojection for layer compositing\n"
        "- TextureFlow: AI video generation from style images (remote Modal GPU endpoint)\n"
        "- Kling v3: Image-to-video generation (3-15s cinematic videos via FAL)\n"
        "- Nano Banana Pro: Text-to-image and image editing (via FAL)\n\n"
        "Typical workflow: detect surface → prepare surface → generate content (e.g. texture_flow) → "
        "subtract background → (optionally segment into islands → reproject)\n\n"
        "All image/video inputs accept URLs or local file paths.\n"
        "All outputs include the processed file and an info dict with metadata."
    ),
)

# Import tools module to register all @mcp.tool() decorated functions.
# This must happen after `mcp` is defined since tools.py imports `mcp` from here.
from . import tools as _tools  # noqa: E402, F401
from . import fal_tools as _fal_tools  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Filter out disabled tools based on config
# ---------------------------------------------------------------------------


def _apply_tool_config() -> None:
    """Remove disabled tools from the server's tool registry."""
    for name, enabled in TOOL_CONFIG.items():
        if not enabled:
            try:
                mcp.remove_tool(name)
            except (KeyError, ValueError):
                pass


_apply_tool_config()
