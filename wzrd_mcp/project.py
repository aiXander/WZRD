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
    return {
        "project": active_project(),
        "project_dir": str(root),
        "exists": root.is_dir(),
        "subdirs": {
            k: str(root / k)
            for k in (SURFACES, GENERATED, VIDEO, REGIONS, CAPTURES, PACKS)
        },
    }


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
