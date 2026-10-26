"""Named project assets — `<project>/assets.json` (show-setup contract A1, G2/G3).

A show names its key files once — the lit photo, the rectified lit crop, the
dark surface, the segmentation map — and every tool takes them as
`asset:<name>` (expanded by `logged_tool` before the body runs, so records
and bodies see real paths). The UI shows the same names in its pickers.

    {"schema": 1, "assets": {
      "surface": {"path": "/abs/file.png", "id": "r_…", "fp": "<size>-<sha1>",
                  "kind": "image", "note": "", "pinned": "2026-10-16T14:30:12",
                  "by": "app"}}}

One name -> one file. Written only by the hub (`/api/project/assets`, the
app's pins), temp + rename under a lock. Pinning indexes the file, so the
render id and fingerprint are stored and a moved file is followed:
resolution = the stored path if it is on disk, else the render id's / the
fingerprint's current file in `library.db`, else not found. A show copied
with "Save as" keeps the source show's absolute paths; resolution prefers the
same relative path inside *this* show when it exists.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import project

ASSETS_FILE = "assets.json"
SCHEMA = 1
PREFIX = "asset:"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")

#: Well-known asset names -> what each is for (the wizard pins these).
ROLES: dict[str, str] = {
    "photo_lit": "venue photo with the projector showing full white — detection input",
    "photo_ambient": "the same view with the room lit — colour/detail source for the surface",
    "surface_lit": "rectified lit crop at the canvas — shape/ControlNet input, segmentation input",
    "surface": "dark colour-corrected surface at the canvas — nano_banana image_1, subtraction background",
    "surface_ambient": "rectified, aligned ambient photo, undarkened",
    "segmentation": "flat-colour region map — extract_color_regions input",
}

_lock = threading.Lock()


def check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"asset name {name!r}: lowercase letters, digits, _ and -, "
                         f"starting with a letter or digit, max 48 chars")
    return name


def assets_path(name: Optional[str] = None) -> Path:
    return project.project_dir(name) / ASSETS_FILE


def _read(name: Optional[str] = None) -> dict:
    try:
        data = json.loads(assets_path(name).read_text())
    except (OSError, ValueError):
        return {"schema": SCHEMA, "assets": {}}
    if not isinstance(data.get("assets"), dict):
        data["assets"] = {}
    return data


def _write(data: dict, name: Optional[str] = None) -> None:
    path = assets_path(name)
    tmp = path.with_name(f".{ASSETS_FILE}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def _in_this_show(stored: Path, show_dir: Path) -> Optional[Path]:
    """A path into another show (a "Save as" copy) -> the same file in this show."""
    root = project.projects_root()
    try:
        rel = stored.relative_to(root)
    except ValueError:
        return None
    if len(rel.parts) < 2 or rel.parts[0] == show_dir.name:
        return None
    local = show_dir.joinpath(*rel.parts[1:])
    return local if local.is_file() else None


def _follow(entry: dict, show_dir: Path) -> tuple[str, bool]:
    from . import library

    stored = Path(entry.get("path") or "")
    if entry.get("path"):
        local = _in_this_show(stored, show_dir)
        if local:
            return str(local), True
        if stored.is_file():
            return str(stored), True
    for col, value in (("render_id", entry.get("id")), ("fp", entry.get("fp"))):
        if not value:
            continue
        rows = library.query(
            f"SELECT path, variant FROM files WHERE {col}=? AND missing=0 ORDER BY variant<>'master', last_seen DESC",
            (value,))
        hits = [r["path"] for r in rows if Path(r["path"]).is_file()]
        if hits:
            mine = [p for p in hits if Path(p).is_relative_to(show_dir)]
            return (mine or hits)[0], True
    return str(stored), False


def _public(name: str, entry: dict, show_dir: Path) -> dict:
    path, exists = _follow(entry, show_dir)
    return {
        "name": name, "path": path, "id": entry.get("id"), "fp": entry.get("fp"),
        "kind": entry.get("kind"), "note": entry.get("note") or "",
        "pinned": entry.get("pinned"), "by": entry.get("by") or "agent", "exists": exists,
    }


def list_assets(name: Optional[str] = None) -> dict[str, dict]:
    """Every named asset of a show, resolved (see the module docstring)."""
    show_dir = project.project_dir(name)
    return {k: _public(k, v, show_dir) for k, v in sorted(_read(name)["assets"].items())}


def resolve(asset: str, name: Optional[str] = None) -> str:
    """`<name>` (or `asset:<name>`) -> the absolute path of that asset's file."""
    key = asset[len(PREFIX):] if asset.startswith(PREFIX) else asset
    show = name or project.active_project()
    entries = _read(show)["assets"]
    if key not in entries:
        known = ", ".join(sorted(entries)) or "none pinned yet"
        raise ValueError(f"show {show!r} has no asset {key!r} (its assets: {known}) — pin it in the app first (Studio ★ or Show setup)")
    path, exists = _follow(entries[key], project.project_dir(show))
    if not exists:
        raise ValueError(f"asset {key!r} of show {show!r} points at {path}, which is gone "
                         f"(not found by render id or fingerprint either) — pin it again")
    return path


def expand_args(kwargs: dict[str, Any], name: Optional[str] = None) -> dict[str, Any]:
    """Swap every `asset:<name>` string — top-level, or inside a list — for its path."""
    def one(v: Any) -> Any:
        return resolve(v, name) if isinstance(v, str) and v.startswith(PREFIX) else v

    out = {}
    for k, v in kwargs.items():
        out[k] = [one(x) for x in v] if isinstance(v, list) else one(v)
    return out


def has_refs(kwargs: dict[str, Any]) -> bool:
    return any(
        (isinstance(v, str) and v.startswith(PREFIX))
        or (isinstance(v, list) and any(isinstance(x, str) and x.startswith(PREFIX) for x in v))
        for v in kwargs.values()
    )


# ---------------------------------------------------------------------------
# Writes (the hub only)
# ---------------------------------------------------------------------------
def pin(asset: str, path: str, *, note: Optional[str] = None, by: str = "agent",
        name: Optional[str] = None) -> dict[str, dict]:
    """Point the asset *asset* at *path* (a media file; `asset:<other>` works too)."""
    from . import identity, library

    check_name(asset)
    show = name or project.active_project()
    show_dir = project.project_dir(show)
    if not show_dir.is_dir():
        raise ValueError(f"no show {show!r} (the app's Show setup creates one)")
    if path.startswith(PREFIX):
        path = resolve(path, show)
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError(f"asset paths are absolute (got {path!r})")
    p = p.resolve()
    if not p.is_file():
        raise ValueError(f"{p} is not a file")
    kind = identity.media_kind(p)
    if kind is None:
        raise ValueError(f"{p.name} is not an image or video")
    it = library.index_file(p) or {}
    with _lock:
        data = _read(show)
        prev = data["assets"].get(asset) or {}
        data["assets"][asset] = {
            "path": str(p), "id": it.get("render_id"), "fp": it.get("fp"), "kind": kind,
            "note": note if note is not None else prev.get("note", ""),
            "pinned": _now(), "by": by,
        }
        data["schema"] = SCHEMA
        _write(data, show)
    return list_assets(show)


def unpin(asset: str, *, name: Optional[str] = None) -> dict[str, dict]:
    """Forget the name (the file stays where it is)."""
    check_name(asset)
    show = name or project.active_project()
    with _lock:
        data = _read(show)
        if data["assets"].pop(asset, None) is not None:
            _write(data, show)
    return list_assets(show)
