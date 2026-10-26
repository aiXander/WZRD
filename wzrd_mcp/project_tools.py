"""Show tools: the app-only setup steps that create a show and bring photos in
(the wizard runs them; agents never see them), and the agent's view of a show."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project
from .server import tool


@tool(title="Set project", tags={"project"}, idempotent=True, app_only=True)
async def set_project(
    name: Optional[str] = None,
    canvas: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Create or open a show (a project folder) and report it; lock a new show's canvas.

    A show is one self-contained folder holding one venue's set — surfaces,
    region masks, layer packs, and the engine's scene.json + effects/ — so
    the realtime engine loads it straight off disk. The canvas is the pixel
    frame every surface, segmentation map, mask and pack is authored in: a
    Nano Banana native size, locked once, never changed afterwards.

    Args:
        name: Show name (created if it doesn't exist yet). Omit to report the show open in the app.
        canvas: New shows only, set once: an aspect ("16:9", "4:3", "1:1", "9:16", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4" → its 1K size) or an exact native size like "2752x1536" (2K).
    """
    if name:
        project.set_project(name)
    if canvas:
        try:
            project.lock_canvas(canvas)
        except ValueError as exc:
            raise ToolError(str(exc))
    return {**project.describe(), "existing_projects": project.list_projects()}


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


@tool(title="Import photos", tags={"project", "surface"}, project_arg=True, app_only=True)
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
        conform: True: cover-crop + resize to the project canvas — for a photo shot from the projector's position that becomes the surface. False: whole frame, full resolution — for the lit/ambient pair going into rectify_surface, whose detection needs every pixel.
    """
    size = project.canvas() if conform else None
    sources = [Path(p).expanduser() for p in paths] if paths else _newest_photos(source_dir, max(1, count))
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise ToolError(f"not found: {', '.join(missing)}")
    photos = await asyncio.gather(*(asyncio.to_thread(_import_one, p, size) for p in sources))
    result: dict = {"photos": list(photos), "canvas": list(size) if size else None}
    if conform and not size:
        result["note"] = "project has no canvas — photos kept their own size (lock one in Show setup)"
    result["_provenance"] = {"parent": {os.path.realpath(p["path"]): p["source"] for p in photos}}
    return result


@tool(title="Show info", tags={"project"}, read_only=True, idempotent=True, project_arg=True)
async def show_info(region: Optional[str] = None, ctx: Optional[Context] = None) -> dict:
    """The show: its canvas, regions (ids for region=, with the box their clips play in), named assets (pass any as "asset:<name>") and folder map.

    The operator sets a show up in the app (photos, surface, regions, pack);
    this is how you read the result.

    Args:
        region: A region id: also return its render inputs — `mask` (white = the region) and `shape` (the surface, black outside the region), both cut to its box, so a render at that aspect lands back on the region exactly.
    """
    from . import assets, provenance

    info = await asyncio.to_thread(project.describe)
    out = {**info, "regions": await asyncio.to_thread(provenance.regions), "roles": assets.ROLES}
    if region:
        out["region_inputs"] = await asyncio.to_thread(provenance.region_inputs, region)
    return out
