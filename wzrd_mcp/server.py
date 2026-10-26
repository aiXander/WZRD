"""FastMCP server composition root for WZRD (FastMCP 4 / MCP SDK 2.x).

At import time:

1. Load `tools_config.json` — the operator gate: per-tool `{enabled, timeout}`.
2. Build the `FastMCP` instance.
3. Import every tool module. Each registers its tools through `tool()` below,
   the one registration path: name = function name, timeout from the config,
   MCP tool annotations, text-only results, logging + error wrapping.
4. Remove the tools the config disables, and refuse to boot if one survives.
5. Render the `instructions` — the always-on context: the physics of additive
   projection, the call contract, and the job map (each job's tools + the
   skill to load first), from the tools that survived the gate. Everything
   deeper (workflows, prompt formulas, model craft) lives in skills and is
   loaded on demand — see `skill_catalog.py`.
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
# Server instance. Its instructions are set at the bottom, once the config
# gate has decided which tools exist.
# ---------------------------------------------------------------------------
mcp = FastMCP("WZRD", version="0.3.0")
APP_ONLY = "app"  # the tag `tool(app_only=True)` sets; hidden from agents below


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
    app_only: bool = False,
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

    `app_only` keeps a tool off the agent's surface: hidden from /mcp and
    refused there, still run by the app through /api/tool. For the
    show-setup steps the operator does by eye in the wizard (corners,
    sliders, region ids) — the agent never needs them.
    """
    if render:
        tags = {*tags, "render"}
    if app_only:
        tags = {*tags, APP_ONLY}

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
def _registered_tools() -> list[Tool]:
    # Sync read of the provider's component dict (what its async `_list_tools`
    # returns): this runs at import, possibly inside a running event loop.
    return [c for c in mcp.local_provider._components.values() if isinstance(c, Tool)]


def _registered_tool_names() -> set[str]:
    return {t.name for t in _registered_tools()}


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
mcp.disable(tags={APP_ONLY})


# ---------------------------------------------------------------------------
# Instructions = level 1 of progressive disclosure: always loaded, so short.
# Clients like Claude Code defer tool schemas and show the model only tool
# NAMES until it loads one, so the job map below is how it knows which tool
# (and which skill) a job needs. Disabled and app-only tools drop out; a tool or
# skill the map doesn't name is a boot warning, an unknown skill a failure.
# ---------------------------------------------------------------------------
_PREAMBLE = """\
WZRD turns physical surfaces (buildings, rocks, trees, stages) into animated light by additive \
projection mapping. A projector only ADDS light: black = no light = the real surface shows, dark \
subjects vanish, so content must be bright and luminous; areas that reflect nothing (sky, glass, \
voids) stay dark whatever you send. The real surface never moves: generated media keeps the \
background pixel-static (no camera motion, no global flicker).

How calls work: files are local and every tool returns absolute paths that chain into the next \
call (inputs also take URLs). A show is a project folder: pass project=<name> on every project tool \
(nothing is remembered between calls; omitted = the show open in the WZRD app). Any media argument \
takes "asset:<name>" (the show's pinned files, show_info). Generators write to the media library \
~/WZRD/media unless dest="project" or "project/<folder>". Look at what you made (view_media) before \
presenting it and let the user judge; at most 6 generations in parallel.

Jobs → tools. Before a job, load its skill with load_skill(name): the workflow, prompt formulas and \
settings live there; tool docs hold only the contract."""

# (job, skills to load first, tools)
_JOB_MAP: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("The show the operator set up in the app (canvas, regions, named assets) and a venue day's "
     "clip library + pre-show checklist", ("run-sheet",), ("show_info",)),
    ("Separate animation per region (windows, panels, trunk vs canopy)", ("region-animation",),
     ("show_info", "texture_flow")),
    ("Images: keyframes on the surface, masks", ("image-prompting",), ("nano_banana",)),
    ("A story or creature animated on the surface", ("projection-story", "video-prompting"),
     ("kling_v25_image_to_video", "extract_frames", "concat_videos", "subtract_background_video")),
    ("Abstract looping textures, VJ loops, region fills, style recipes", ("textureflow",),
     ("texture_flow", "list_recipes", "save_recipe")),
    ("Check a clip and make it engine-playable", (), ("view_media", "check_frames", "to_hap")),
    ("The operator's Studio selection (\"these styles\", \"these clips\") and render jobs", (),
     ("studio_selection", "jobs")),
    ("Change what the engine renders now (the design leg; promoting to live is the operator's; "
     "get_scene_context first)",
     ("live-engine",),
     ("get_scene_context", "upsert_binding", "remove_binding", "upsert_effect", "set_groups",
      "set_labels", "get_preview")),
]


def _render_instructions() -> str:
    alive = {t.name for t in _registered_tools() if APP_ONLY not in (t.tags or set())}  # what agents see
    skills = skill_catalog.catalog()
    lines, mapped = [], {"load_skill"}
    for job, job_skills, tools in _JOB_MAP:
        unknown = [s for s in job_skills if s not in skills]
        if unknown:
            raise RuntimeError(f"server._JOB_MAP names unknown skills {unknown}")
        mapped.update(tools)
        present = [t for t in tools if t in alive]
        if present or job_skills:
            skill = f" [load {' + '.join(job_skills)}]" if job_skills else ""
            lines.append(f"- {job}{skill}" + (f": {', '.join(present)}" if present else ""))
    if unmapped := sorted(alive - mapped):
        _emit(f"\033[33m⚠ WZRD: tools missing from server._JOB_MAP: {unmapped}\033[0m")
    named = {s for _, job_skills, _ in _JOB_MAP for s in job_skills}
    if unnamed := sorted(set(skills) - named):
        _emit(f"\033[33m⚠ WZRD: skills missing from server._JOB_MAP: {unnamed}\033[0m")
    return _PREAMBLE + "\n" + "\n".join(lines)


mcp.instructions = _render_instructions()
