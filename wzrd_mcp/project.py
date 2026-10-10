"""Project-local file store — the local replacement for the S3 publish path.

Every tool used to write into `/tmp` and then hand the file to
`file_io.upload()`, which pushed it to S3 and returned a public URL. That made
the offline pipeline unusable without cloud credentials, and left the engine
unable to consume its own layer packs (render-core reads masks off local disk).

Here, outputs land in a **project folder** on this machine and tools return
absolute local paths. Since `file_io.resolve_input()` already accepts local
paths, chaining tools keeps working unchanged.

Layout (created lazily, per project)::

    projects/<name>/
      project.json        locked canvas + spend cap (C1; absent = legacy project)
      scene.json          engine entry point (authored by the engine/UI)
      effects/            project-local WGSL effects   <- render-core convention
      packs/<pack>/       layer packs (pack.json + masks/ + references/)
      surfaces/           prepared surface stills + alignment aids
      generated/          model outputs (FAL, TextureFlow, …)
      video/              processed video (subtract / reproject / simulate)
      regions/            extract_color_regions runs (one subdir per run)
      captures/           camera snapshots

`packs/` and `effects/` sit next to `scene.json` deliberately: render-core
resolves a scene's `pack` relative to the scene file and looks for effects in
`<scene_dir>/effects/`, so a project folder *is* a scene directory.

The active project comes from (in order): `set_project()` during this session,
the `WZRD_PROJECT` env var, else `"default"`. Override the root that projects
live under with `WZRD_PROJECTS_ROOT`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Output kind -> subdirectory. Passing an unknown kind just creates that dir.
SURFACES = "surfaces"
GENERATED = "generated"
VIDEO = "video"
REGIONS = "regions"
CAPTURES = "captures"
PACKS = "packs"

_DEFAULT_PROJECT = "default"
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

# Session-level active project, set via the `set_project` MCP tool.
_active_project: str | None = None


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
def projects_root() -> Path:
    """Directory holding all projects (repo-local `projects/`, gitignored)."""
    env = os.getenv("WZRD_PROJECTS_ROOT")
    root = Path(env).expanduser() if env else _REPO_ROOT / "projects"
    return root.resolve()


def slugify(name: str) -> str:
    """Reduce a free-form project name to a safe single path segment."""
    slug = _SLUG_RE.sub("-", name.strip()).strip("-._")
    if not slug:
        raise ValueError(f"project name {name!r} has no usable characters")
    return slug


def active_project() -> str:
    """Name of the project new outputs are written to."""
    if _active_project:
        return _active_project
    env = os.getenv("WZRD_PROJECT")
    return slugify(env) if env else _DEFAULT_PROJECT


def set_project(name: str) -> Path:
    """Make *name* the active project for this session, creating it if needed."""
    global _active_project
    _active_project = slugify(name)
    path = project_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_dir(name: str | None = None) -> Path:
    """Absolute path of a project folder (not created)."""
    return projects_root() / (slugify(name) if name else active_project())


def kind_dir(kind: str) -> Path:
    """Absolute path of one output subdirectory, created on demand."""
    path = project_dir() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_projects() -> list[str]:
    """Existing project names, newest-modified first."""
    root = projects_root()
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return [p.name for p in sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)]


def describe() -> dict:
    """Summary of the active project — what the `set_project` tool reports back."""
    root = project_dir()
    meta = read_project_file()
    return {
        "project": active_project(),
        "project_dir": str(root),
        "exists": root.is_dir(),
        "canvas": meta.get("canvas") if meta else None,
        "spend_cap_usd": meta.get("spend_cap_usd") if meta else None,
        "subdirs": {
            k: str(root / k)
            for k in (SURFACES, GENERATED, VIDEO, REGIONS, CAPTURES, PACKS)
        },
    }


# ---------------------------------------------------------------------------
# project.json — the locked canvas (C1, D7)
# ---------------------------------------------------------------------------
PROJECT_FILE = "project.json"
PROJECT_SCHEMA = 1
DEFAULT_SPEND_CAP_USD = 20.0
_TIERS = {1: "1K", 2: "2K", 4: "4K"}


def native_canvases() -> dict[str, tuple[int, int]]:
    """Every allowed canvas: Nano Banana's native sizes, "WxH" -> (w, h)."""
    from wzrd.utils import NANO_BANANA_SIZES

    return {
        f"{w * k}x{h * k}": (w * k, h * k)
        for (w, h) in NANO_BANANA_SIZES.values() for k in _TIERS
    }


def parse_canvas(value: str) -> tuple[int, int]:
    """'16:9' (that aspect's 1K size) or 'WxH' (must be a native size) -> (w, h)."""
    from wzrd.utils import NANO_BANANA_SIZES

    v = value.strip().lower().replace(" ", "")
    if v in NANO_BANANA_SIZES:
        return NANO_BANANA_SIZES[v]
    sizes = native_canvases()
    if v in sizes:
        return sizes[v]
    names = ", ".join(f"{k} ({w}x{h})" for k, (w, h) in NANO_BANANA_SIZES.items())
    raise ValueError(
        f"canvas {value!r} is not a Nano Banana native size. Give an aspect — {names} — "
        f"or one of those sizes doubled/quadrupled (2K/4K), e.g. '2752x1536'."
    )


def canvas_aspect(size: tuple[int, int]) -> tuple[str, str]:
    """(aspect name, resolution tier) of a native canvas, e.g. ('16:9', '2K')."""
    from wzrd.utils import NANO_BANANA_SIZES

    for name, (w, h) in NANO_BANANA_SIZES.items():
        for k, tier in _TIERS.items():
            if (w * k, h * k) == tuple(size):
                return name, tier
    raise ValueError(f"{size[0]}x{size[1]} is not a native canvas")


def project_file(name: str | None = None) -> Path:
    return project_dir(name) / PROJECT_FILE


def read_project_file(name: str | None = None) -> dict | None:
    """The project's `project.json`, or None for a legacy project."""
    path = project_file(name)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_project_file(data: dict, name: str | None = None) -> None:
    """Atomic write (temp file + rename) — the app may read it any time."""
    path = project_file(name)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def canvas() -> tuple[int, int] | None:
    """The active project's locked canvas (w, h), or None (legacy project)."""
    meta = read_project_file()
    if meta and meta.get("canvas"):
        w, h = meta["canvas"]
        return int(w), int(h)
    return None


def _has_content(root: Path) -> bool:
    return any(p.is_file() and not p.name.startswith(".") for p in root.rglob("*"))


def lock_canvas(value: str) -> tuple[int, int]:
    """Set the active project's canvas once (C1). Same value again is a no-op;
    a different one, or a canvas for a project that already holds files
    without one (legacy), is an error."""
    size = parse_canvas(value)
    root = project_dir()
    meta = read_project_file()
    if meta and meta.get("canvas"):
        current = tuple(meta["canvas"])
        if current != size:
            raise ValueError(
                f"project {active_project()!r} is locked to canvas {current[0]}x{current[1]}; "
                f"a canvas never changes — start a new project for {size[0]}x{size[1]}"
            )
        return size
    if meta is None and root.is_dir() and _has_content(root):
        raise ValueError(
            f"project {active_project()!r} already holds files made without a canvas (a legacy "
            f"project); a canvas is chosen once, for new projects only — start a new project"
        )
    root.mkdir(parents=True, exist_ok=True)
    data = {"schema": PROJECT_SCHEMA, "canvas": list(size),
            "spend_cap_usd": DEFAULT_SPEND_CAP_USD, **(meta or {})}
    data["canvas"] = list(size)
    write_project_file(data)
    return size


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _unique(directory: Path, stem: str, ext: str) -> Path:
    """First free `<stem><ext>`, `<stem>_2<ext>`, … in *directory*."""
    candidate = directory / f"{stem}{ext}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{ext}"
        n += 1
    return candidate


def output_path(tool: str, ext: str, kind: str = GENERATED, name: str = "") -> str:
    """Reserve a fresh output file path inside the active project.

    Tools write straight to this path, so nothing has to be copied afterwards.
    Libraries that emit sidecar files next to their output (the alignment-aid
    video, for instance) land in the project folder for free.
    """
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    stem = slugify(name) if name else f"{tool}_{_stamp()}"
    return str(_unique(kind_dir(kind), stem, ext))


def output_dir(tool: str, kind: str = GENERATED, name: str = "") -> str:
    """Reserve a fresh output *directory* inside the active project."""
    stem = slugify(name) if name else f"{tool}_{_stamp()}"
    path = _unique(kind_dir(kind), stem, "")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# ---------------------------------------------------------------------------
# Storing files that were produced elsewhere
# ---------------------------------------------------------------------------
def store(
    src: str,
    tool: str = "file",
    kind: str = GENERATED,
    ext: str = "",
    name: str = "",
    move: bool | None = None,
) -> str:
    """Place an existing file into the active project; returns its new path.

    Used for files WZRD did not write directly — a download from FAL, a camera
    frame, a library output that chose its own path.

    `move=None` (default) moves the file when it sits in the system temp dir
    and copies otherwise, so a caller's own inputs are never consumed.
    """
    source = Path(src).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"cannot store missing file: {source}")

    if move is None:
        move = _is_temp(source)

    dest = Path(output_path(tool, ext or source.suffix, kind=kind, name=name))
    if move:
        shutil.move(str(source), dest)
    else:
        shutil.copy2(source, dest)
    return str(dest)


def _is_temp(path: Path) -> bool:
    try:
        return path.is_relative_to(Path(tempfile.gettempdir()).resolve())
    except AttributeError:  # pragma: no cover — Python < 3.9
        return str(path).startswith(tempfile.gettempdir())


# ---------------------------------------------------------------------------
# Downloading remote results straight into the project
# ---------------------------------------------------------------------------
async def download_async(
    url: str,
    tool: str = "download",
    kind: str = GENERATED,
    ext: str = "",
    name: str = "",
) -> str:
    """Stream a remote result into the active project; returns the local path.

    This is the landing point for every remote tool call (FAL, TextureFlow):
    results come home to disk instead of being re-hosted.
    """
    import httpx

    if not ext:
        from urllib.parse import urlparse

        ext = Path(urlparse(url).path).suffix

    dest = Path(output_path(tool, ext, kind=kind, name=name))
    async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                async for chunk in resp.aiter_bytes():
                    fh.write(chunk)
    return str(dest)


# ---------------------------------------------------------------------------
# Async wrappers (file moves/copies can be slow for video)
# ---------------------------------------------------------------------------
async def store_async(
    src: str,
    tool: str = "file",
    kind: str = GENERATED,
    ext: str = "",
    name: str = "",
    move: bool | None = None,
) -> str:
    """Async version of `store` — offloads the copy/move to a thread."""
    return await asyncio.to_thread(store, src, tool, kind, ext, name, move)
