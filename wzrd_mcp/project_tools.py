"""MCP tools for choosing where output lands — the project folder."""

from __future__ import annotations

from typing import Optional

from fastmcp import Context

from . import project
from .server import tool


@tool(title="Set project", tags={"project"}, idempotent=True)
async def set_project(name: Optional[str] = None, ctx: Optional[Context] = None) -> dict:
    """Choose the project folder that every subsequent tool writes its output into.

    A project is one self-contained folder on this machine holding everything
    for a single mapping job — prepared surfaces, generated content, region
    masks, layer packs, and the engine's scene.json + effects/ — so a session's
    files stay together and the realtime engine can load them straight off disk.

    Call this FIRST in any content-pipeline session. Without it, output goes to
    the project named by the WZRD_PROJECT env var, else "default".

    Args:
        name: Project name (created if it doesn't exist yet). Omit to
            just report the active project and list the existing ones.
    """
    if name:
        project.set_project(name)
    return {
        **project.describe(),
        "projects_root": str(project.projects_root()),
        "existing_projects": project.list_projects(),
    }
