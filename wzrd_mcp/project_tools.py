"""MCP tools for choosing where output lands — the project folder."""

from __future__ import annotations

import time
from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project
from ._log import log_done, log_error, logged_tool
from .server import mcp, get_timeout


@mcp.tool(timeout=get_timeout("set_project"))
@logged_tool
async def set_project(name: str = "", ctx: Optional[Context] = None) -> dict:
    """Choose the project folder that every subsequent tool writes its output into.

    A project is one self-contained folder on this machine holding everything
    for a single mapping job — prepared surfaces, generated content, region
    masks, layer packs, and the engine's scene.json + effects/ — so a session's
    files stay together and the realtime engine can load them straight off disk.

    Call this FIRST in any content-pipeline session. Without it, output goes to
    the project named by the WZRD_PROJECT env var, else "default".

    Args:
        name: Project name (created if it doesn't exist yet). Leave empty to
            just report the active project and list the existing ones.
    """
    _name = "set_project"
    t0 = time.time()
    try:
        if name:
            project.set_project(name)

        result = {
            **project.describe(),
            "projects_root": str(project.projects_root()),
            "existing_projects": project.list_projects(),
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))
