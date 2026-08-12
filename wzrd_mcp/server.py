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
    "set_project": {"enabled": True, "timeout": 30},
    "subtract_background_frame": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "subtract_background_video": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "detect_projection_surface": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "align_images": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "darken_surface": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "prepare_surface": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "extract_color_regions": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "reproject_video": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    # Off until it is rehomed out of Eden's Modal workspace — see
    # docs/TODO/eden-decoupling.md.
    "texture_flow": {"enabled": False, "timeout": 1500},
    "kling_v3_image_to_video": {"enabled": True, "timeout": 600},
    "nano_banana_pro": {"enabled": True, "timeout": _DEFAULT_TIMEOUT},
    "simulate_view": {"enabled": True, "timeout": 300},
    "capture_camera_snapshot": {"enabled": True, "timeout": 30},
    # §5.10 engine authoring tools — localhost-only (the engine WS binds
    # 127.0.0.1:9123), so they default OFF; the local tools_config.json
    # flips them on. The Modal image also never installs `websockets`, so a
    # cloud deployment cannot carry them regardless of config.
    "get_scene_context": {"enabled": False, "timeout": 60},
    "upsert_binding": {"enabled": False, "timeout": 30},
    "remove_binding": {"enabled": False, "timeout": 30},
    "upsert_effect": {"enabled": False, "timeout": 30},
    "remove_effect": {"enabled": False, "timeout": 30},
    "set_groups": {"enabled": False, "timeout": 30},
    "set_labels": {"enabled": False, "timeout": 30},
    "set_scene": {"enabled": False, "timeout": 30},
    "validate_wgsl": {"enabled": False, "timeout": 30},
    "get_preview": {"enabled": False, "timeout": 30},
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
        "WZRD is a VJ projection-mapping toolkit. It spans two areas — load the\n"
        "tools for whichever the task needs; tool names are self-describing.\n\n"
        "1) OFFLINE CONTENT PIPELINE — prepare surfaces and generate/segment the\n"
        "content that becomes a layer pack. Load these when the task is about\n"
        "producing assets from photos or prompts (not driving the live engine).\n"
        "Everything is LOCAL: call set_project FIRST to pick the folder this\n"
        "session writes into; every tool then returns absolute local paths (and\n"
        "accepts them as inputs), so results are chained path-to-path and land\n"
        "next to the scene the engine loads.\n"
        "- prepare_surface / extract_color_regions / build_layerpack — surface prep,\n"
        "  color-region segmentation (islands), and authoring the layer pack.\n"
        "- subtract_background_video / reproject_video / simulate_view — extract\n"
        "  moving content, reposition island regions, preview additive projection.\n"
        "- kling_v25_image_to_video (5/10s image→video via FAL), nano_banana_pro\n"
        "  (text→image + multi-reference edit via FAL). Remote results download\n"
        "  into the project; local paths given as inputs are uploaded for you.\n"
        "Typical flow: set_project → prepare_surface → generate content (e.g.\n"
        "nano_banana_pro → kling_v25_image_to_video) → subtract_background_video →\n"
        "(optionally extract_color_regions → reproject_video) → build_layerpack.\n\n"
        "2) LIVE ENGINE AUTHORING — drive the realtime render-core over its WS\n"
        "(ws://127.0.0.1:9123). Load these when the operator wants to change what's\n"
        "rendering NOW (bindings, WGSL effects, layer labels/groups). ALWAYS call\n"
        "get_scene_context FIRST — it reflects the human's live UI edits and tells\n"
        "you what exists before you mutate:\n"
        "- get_scene_context / get_preview — read live state; grab a design-composite frame.\n"
        "- upsert_binding / remove_binding — add/replace/remove one binding (granular,\n"
        "  CAS-guarded). set_scene only for initial authoring or a full structural rewrite.\n"
        "- upsert_effect / remove_effect / validate_wgsl — author project-local WGSL\n"
        "  effects (naga-validated, pre-flight probed; validate_wgsl is a cheap dry run).\n"
        "- set_labels / set_groups — name layers and define groups so later commands\n"
        "  can target the operator's surface-language ('the trunk', group 'canopy').\n"
        "Writes target the DESIGN leg only; promoting design→live is a human UI act.\n"
        "Requires the local `.[engine]` extra (websockets); absent on Modal.\n\n"
        "All image/video inputs accept local file paths or URLs.\n"
        "All outputs include the produced file's local path and an info dict."
    ),
)

# Import tools module to register all @mcp.tool() decorated functions.
# This must happen after `mcp` is defined since tools.py imports `mcp` from here.
from . import project_tools as _project_tools  # noqa: E402, F401
from . import tools as _tools  # noqa: E402, F401
from . import fal_tools as _fal_tools  # noqa: E402, F401
from . import local_tools as _local_tools  # noqa: E402, F401
from . import engine_tools as _engine_tools  # noqa: E402, F401  (§5.10 — no-op without `websockets`)

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
