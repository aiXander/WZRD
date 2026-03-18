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
_DEFAULT_TIMEOUT = 120  # seconds

_DEFAULT_TOOLS = {
    "subtract_background_frame": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "subtract_background_video": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "detect_projection_surface": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "align_images": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "darken_surface": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "prepare_surface": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "extract_color_regions": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "reproject_video": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "texture_flow": {"enabled": True, "timeout": 1500},
    "kling_v3_image_to_video": {"enabled": True, "timeout": 600},
    "nano_banana_pro": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "simulate_view": {"enabled": True, "timeout": 300},
    "capture_camera_snapshot": {"enabled": True, "timeout": 30},
}


def _load_tool_config() -> dict:
    """Load tool config. Supports both old (bool) and new (dict) formats."""
    merged = {k: dict(v) for k, v in _DEFAULT_TOOLS.items()}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            raw = json.load(f)
        for name, value in raw.items():
            if isinstance(value, bool):
                # Legacy format: bare bool → convert
                merged.setdefault(name, {"enabled": True, "timeout": _DEFAULT_TIMEOUT})
                merged[name]["enabled"] = value
            elif isinstance(value, dict):
                merged.setdefault(name, {"enabled": True, "timeout": _DEFAULT_TIMEOUT})
                merged[name].update(value)
    return merged


TOOL_CONFIG = _load_tool_config()


def get_timeout(tool_name: str) -> float:
    """Get the configured timeout (seconds) for a tool."""
    cfg = TOOL_CONFIG.get(tool_name)
    if cfg is None:
        print(f"\033[33m⚠ WZRD: tool '{tool_name}' not found in tools_config.json — using default timeout ({_DEFAULT_TIMEOUT}s)\033[0m")
        return float(_DEFAULT_TIMEOUT)
    return float(cfg.get("timeout", _DEFAULT_TIMEOUT))

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
from . import local_tools as _local_tools  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Filter out disabled tools based on config
# ---------------------------------------------------------------------------


def _apply_tool_config() -> None:
    """Remove disabled tools from the server's tool registry."""
    for name, cfg in TOOL_CONFIG.items():
        if not cfg.get("enabled", True):
            try:
                mcp.local_provider.remove_tool(name)
            except (KeyError, ValueError):
                pass


_apply_tool_config()
