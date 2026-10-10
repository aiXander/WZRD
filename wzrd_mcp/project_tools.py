"""MCP tools for choosing where output lands — the project folder."""

from __future__ import annotations

from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project
from .server import tool


@tool(title="Set project", tags={"project"}, idempotent=True)
async def set_project(
    name: Optional[str] = None,
    canvas: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Choose the project folder that every subsequent tool writes its output into.

    A project is one self-contained folder on this machine holding everything
    for a single mapping job — prepared surfaces, generated content, region
    masks, layer packs, and the engine's scene.json + effects/ — so a session's
    files stay together and the realtime engine can load them straight off disk.

    Call this FIRST in any content-pipeline session. Without it, output goes to
    the project named by the WZRD_PROJECT env var, else "default".

    A new project should lock its canvas here, once: the pixel frame every
    surface, segmentation map, mask and pack is authored in. Pick the Nano
    Banana native size nearest the projector (a 16:9 projector → "16:9" =
    1376x768); Nano Banana then answers surface edits at exactly the canvas,
    so nothing is ever cropped or stretched. Surface tools conform photos to it
    and build_layerpack authors at it. It can never change afterwards.

    Args:
        name: Project name (created if it doesn't exist yet). Omit to just report the active project and list the existing ones.
        canvas: New projects only, set once: an aspect ("16:9", "4:3", "1:1", "9:16", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4" → its 1K size) or an exact native size like "2752x1536" (2K).
    """
    if name:
        project.set_project(name)
    if canvas:
        try:
            project.lock_canvas(canvas)
        except ValueError as exc:
            raise ToolError(str(exc))
    return {
        **project.describe(),
        "projects_root": str(project.projects_root()),
        "existing_projects": project.list_projects(),
    }
