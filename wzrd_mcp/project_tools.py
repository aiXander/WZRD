"""MCP tools for choosing where output lands — the project folder."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project
from .server import tool


@tool(title="Set project", tags={"project"}, idempotent=True)
async def set_project(
    name: Optional[str] = None,
    canvas: Optional[str] = None,
    spend_cap_usd: Optional[float] = None,
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

    Paid tools (nano_banana, kling_*, texture_flow) count against the
    project's soft spend cap (default $20): every result reports the running
    spend, warns at 80 %, and a call past the cap is refused.

    Args:
        name: Project name (created if it doesn't exist yet). Omit to just report the active project and list the existing ones.
        spend_cap_usd: Change the project's spend cap — only when Xander asks for it.
        canvas: New projects only, set once: an aspect ("16:9", "4:3", "1:1", "9:16", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4" → its 1K size) or an exact native size like "2752x1536" (2K).
    """
    if name:
        project.set_project(name)
    try:
        if canvas:
            project.lock_canvas(canvas)
        if spend_cap_usd is not None:
            project.set_spend_cap(spend_cap_usd)
    except ValueError as exc:
        raise ToolError(str(exc))
    from . import provenance

    return {
        **project.describe(),
        "spent_usd": round(provenance.spent(), 2),
        "projects_root": str(project.projects_root()),
        "existing_projects": project.list_projects(),
    }


def _newest_photos(folder: str, count: int) -> list[Path]:
    from wzrd.utils import PHOTO_EXTS

    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ToolError(f"source_dir {root} is not a folder")

    def received(p: Path) -> float:  # AirDrop keeps the shot's mtime; birthtime = arrival
        st = p.stat()
        return max(st.st_mtime, getattr(st, "st_birthtime", 0.0))

    photos = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_EXTS
              and not p.name.startswith(".")]
    if not photos:
        raise ToolError(f"no photos (jpg/png/webp/heic/tif) in {root}")
    return sorted(photos, key=received, reverse=True)[:count]


def _import_one(src: Path, size: Optional[tuple[int, int]]) -> dict:
    from wzrd.utils import conform_to_size, load_upright

    img = load_upright(src).convert("RGB")
    shot = img.size
    if size:
        img = conform_to_size(img, size)
    out = project.output_path("import_photos", ".jpg", kind=project.CAPTURES, name=src.stem)
    img.save(out, quality=95)
    return {"path": out, "source": str(src), "shot_size": list(shot), "size": list(img.size)}


@tool(title="Import photos", tags={"project", "surface"})
async def import_photos(
    paths: Optional[list[str]] = None,
    count: int = 1,
    source_dir: str = "~/Downloads",
    conform: bool = True,
    ctx: Optional[Context] = None,
) -> dict:
    """Bring photos (iPhone HEIC included) into the project's captures/ — upright, and by default at the project canvas.

    Without `paths`, takes the `count` newest photos in `source_dir` (AirDrop
    lands them in ~/Downloads). Every photo is decoded and turned the way it
    was shot (EXIF rotation), then saved as JPEG.

    Args:
        paths: Photo files to import. Omit to take the newest ones from source_dir.
        count: How many of the newest photos to take when `paths` is omitted.
        source_dir: Folder to take the newest photos from.
        conform: True: cover-crop + resize to the project canvas — for a photo shot from the projector's position that becomes the surface. False: whole frame, full resolution — for night/day pairs going into prepare_surface(day_image=…), whose detection needs every pixel.
    """
    size = project.canvas() if conform else None
    sources = [Path(p).expanduser() for p in paths] if paths else _newest_photos(source_dir, max(1, count))
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise ToolError(f"not found: {', '.join(missing)}")
    photos = await asyncio.gather(*(asyncio.to_thread(_import_one, p, size) for p in sources))
    result: dict = {"photos": list(photos), "canvas": list(size) if size else None}
    if conform and not size:
        result["note"] = "project has no canvas — photos kept their own size (set_project(canvas=…) on a new project)"
    result["_provenance"] = {"parent": {os.path.realpath(p["path"]): p["source"] for p in photos}}
    return result
