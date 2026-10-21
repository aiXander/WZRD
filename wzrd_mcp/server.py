"""FastMCP server composition root for WZRD (FastMCP 4 / MCP SDK 2.x).

At import time:

1. Load `tools_config.json` — the operator gate: per-tool `{enabled, timeout}`.
2. Build the `FastMCP` instance. Its `instructions` are the always-on context:
   the physics of additive projection, the local-first file contract, and the
   one-line skills index. Everything deeper (workflows, prompt formulas, model
   craft) lives in skills and is loaded on demand — see `skill_catalog.py`.
3. Import every tool module. Each registers its tools through `tool()` below,
   the one registration path: name = function name, timeout from the config,
   MCP tool annotations, text-only results, logging + error wrapping.
4. Remove the tools the config disables, and refuse to boot if one survives.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.tools import Tool
from mcp.types import ToolAnnotations

from . import skill_catalog
from ._log import _emit, logged_tool
from ._text_result import jsonify_result

# ---------------------------------------------------------------------------
# tools_config.json — the single source of per-tool toggles + timeouts.
# A registered tool missing from it runs enabled with the default timeout,
# and says so on stderr at boot.
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "tools_config.json"
_DEFAULT_TIMEOUT = 120  # seconds

TOOL_CONFIG: dict[str, dict] = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.exists() else {}


def get_timeout(tool_name: str) -> float:
    cfg = TOOL_CONFIG.get(tool_name)
    if cfg is None:
        _emit(f"\033[33m⚠ WZRD: tool '{tool_name}' has no tools_config.json entry — "
              f"enabled, default timeout {_DEFAULT_TIMEOUT}s\033[0m")
        return float(_DEFAULT_TIMEOUT)
    return float(cfg.get("timeout", _DEFAULT_TIMEOUT))


# ---------------------------------------------------------------------------
# Server instance. Instructions = level 1 of progressive disclosure: short,
# always loaded, and pointing at the skills for everything else.
# ---------------------------------------------------------------------------
_INSTRUCTIONS = f"""\
WZRD turns physical surfaces (buildings, rocks, trees, walls, stages) into animated light: \
additive projection mapping, where only what changes gets projected and the rest stays dark \
and merges with reality.

Physics that governs every decision:
- A projector only ADDS light onto a (dark) surface. Black output = no light = the real surface shows.
- Content must be bright and luminous to read; dark subjects vanish. Light spill onto nearby \
surface features sells the illusion.
- Areas that reflect nothing (sky, glass, voids, steep angles) stay dark whatever you send there.
- The real surface never moves: generated media must keep the background pixel-static (no camera \
motion, no global flicker), or background subtraction lights up the whole frame.

Files are LOCAL and every tool returns absolute paths, which chain straight into the next tool. \
Inputs accept local paths or URLs. A show is a project folder: set_project creates/opens one, then \
pass project=<name> on every venue, region= and generator call (nothing is remembered between \
calls; omitted = the show open in the WZRD app). Generators (nano_banana, texture_flow, kling_*, \
to_hap, extract_frames, concat_videos) write to the media library (~/WZRD/media: the Studio target \
folder, else inbox/); dest="project" (generated/) or dest="project/<folder>" puts one into the \
project instead. A show names its key files (pin_asset; list_assets shows them and the show's \
folder map: captures/ surfaces/ segmentation/ regions/ styles/ packs/ …): pass "asset:<name>" \
(e.g. "asset:surface") as any media argument. \
Look at what you made (view_media) before presenting it; don't celebrate first results, let the \
user judge. Run at most 6 generation calls in parallel.

Skills hold the workflows, prompt formulas and model craft. Before any multi-step job or any \
generation prompt, load the matching skill with load_skill(name) (also readable as the MCP \
resource skill://<name>/SKILL.md):
{skill_catalog.index_lines()}"""

mcp = FastMCP("WZRD", instructions=_INSTRUCTIONS, version="0.3.0")


# ---------------------------------------------------------------------------
# The one tool-registration path.
# ---------------------------------------------------------------------------
def tool(
    *,
    title: str,
    tags: set[str],
    read_only: bool = False,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = False,
    project_arg: bool = False,
    render: bool = False,
    dest_arg: bool = False,
):
    """Register an async tool function with WZRD's standard stack.

    `@mcp.tool(output_schema=None)` → `jsonify_result` (one JSON text block,
    never text + structuredContent) → `logged_tool` (logging, ToolError
    wrapping, request timing, session scope, render records) → body. The
    annotations are MCP's behaviour hints: `open_world` = calls a
    paid/remote service, `destructive` = overwrites or deletes user-visible
    state, `read_only` = no side effects.

    `project_arg` adds a `project=` param (a per-call project override,
    Studio S14). `render` makes the tool a generator: it adds `dest=` (where
    outputs land — the library by default), every call becomes a `jobs` row
    (and counts toward the daily spend), and the tag `render` is added.
    `dest_arg` alone adds `dest=` without a job (quick library writers:
    extract_frames, concat_videos).
    """
    if render:
        tags = {*tags, "render"}

    def deco(fn):
        return mcp.tool(
            name=fn.__name__,
            title=title,
            tags=tags,
            annotations=ToolAnnotations(
                title=title,
                read_only_hint=read_only,
                destructive_hint=destructive,
                idempotent_hint=idempotent,
                open_world_hint=open_world,
            ),
            output_schema=None,
            timeout=get_timeout(fn.__name__),
        )(jsonify_result(logged_tool(fn, project_arg=project_arg, render=render, dest_arg=dest_arg)))

    return deco


# Tool modules register via `tool()` on import; they need `mcp`/`tool` above.
from . import project_tools as _project_tools  # noqa: E402, F401
from . import skill_tools as _skill_tools  # noqa: E402, F401
from . import tools as _tools  # noqa: E402, F401
from . import recipe_tools as _recipe_tools  # noqa: E402, F401
from . import media_tools as _media_tools  # noqa: E402, F401
from . import fal_tools as _fal_tools  # noqa: E402, F401
from . import google_tools as _google_tools  # noqa: E402, F401
from . import local_tools as _local_tools  # noqa: E402, F401
from . import studio_tools as _studio_tools  # noqa: E402, F401
from . import engine_tools as _engine_tools  # noqa: E402, F401  (no-op without `websockets`)
from . import hub as _hub  # noqa: E402, F401  (the app's /api/* routes)


# ---------------------------------------------------------------------------
# Operator gate: drop disabled tools; a disabled tool that is still
# registered afterwards is a boot failure, not a warning.
# ---------------------------------------------------------------------------
def _registered_tool_names() -> set[str]:
    # Sync read of the provider's component dict (what its async `_list_tools`
    # returns): this runs at import, possibly inside a running event loop.
    return {c.name for c in mcp.local_provider._components.values() if isinstance(c, Tool)}


def _apply_tool_config() -> None:
    disabled = {name for name, cfg in TOOL_CONFIG.items() if not cfg.get("enabled", True)}
    for name in disabled:
        try:
            mcp.local_provider.remove_tool(name)
        except (KeyError, ValueError, LookupError):
            pass  # not registered (e.g. engine tools without `websockets`)
    alive = disabled & _registered_tool_names()
    if alive:
        raise RuntimeError(f"tools_config.json disables {sorted(alive)} but they are still registered")


_apply_tool_config()
