"""Prompt templates (show-setup contract P1, G7).

Two scopes, one file per template, `<kind>/<name>.json`:

- **WZRD** (every show): `library/prompts/` in this repo (`WZRD_PROMPTS_ROOT`
  overrides; the default is what a Finder-launched app sees, like recipes).
  Shipped: `segmentation/default`, `mask/default`.
- **project**: `<project>/prompts/` — a show's own forks.

    {"schema": 1, "kind": "segmentation", "name": "default", "prompt": "…",
     "notes": null, "updated": "2026-10-16T14:30:12", "from": null}

`{n_regions}` is the one placeholder (plain string replace by the caller —
prompts may hold other braces). `from` names what a template was forked
from (e.g. "wzrd:segmentation/default"), when the writer says. Written by
the hub only (`/api/prompts`, MCP `save_prompt`), temp + rename.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import project

SCHEMA = 1
SCOPES = ("project", "wzrd")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
PLACEHOLDER = "{n_regions}"


def prompts_root() -> Path:
    env = os.getenv("WZRD_PROMPTS_ROOT")
    root = Path(env).expanduser() if env else project._REPO_ROOT / "library" / "prompts"
    return root.resolve()


def _scope_root(scope: str, show: Optional[str]) -> Path:
    if scope == "wzrd":
        return prompts_root()
    if scope == "project":
        return project.project_dir(show) / project.PROMPTS
    raise ValueError(f"scope must be 'project' or 'wzrd' (got {scope!r})")


def _slug(value: str, what: str) -> str:
    if not isinstance(value, str) or not SLUG_RE.match(value):
        raise ValueError(f"prompt {what} {value!r}: lowercase letters, digits, _ and -, max 48 chars")
    return value


def _load(path: Path, scope: str) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return {
        "kind": data.get("kind") or path.parent.name, "name": data.get("name") or path.stem,
        "scope": scope, "prompt": data.get("prompt") or "", "notes": data.get("notes"),
        "updated": data.get("updated"), "path": str(path), "from": data.get("from"),
    }


def list_prompts(kind: Optional[str] = None, show: Optional[str] = None) -> list[dict]:
    """Templates of *kind* (all kinds when None): project scope first, then WZRD;
    `default` first within a scope, then by name."""
    if kind:
        _slug(kind, "kind")
    out: list[dict] = []
    for scope in SCOPES:
        root = _scope_root(scope, show)
        if not root.is_dir():
            continue
        folders = [root / kind] if kind else sorted(p for p in root.iterdir() if p.is_dir())
        for folder in folders:
            files = sorted(folder.glob("*.json"), key=lambda p: (p.stem != "default", p.stem))
            out += [p for p in (_load(f, scope) for f in files) if p]
    return out


def save_prompt(kind: str, name: str, prompt: str, scope: str = "project", show: Optional[str] = None,
                notes: Optional[str] = None, origin: Optional[str] = None) -> dict:
    """Write one template (an existing one of that scope/kind/name is replaced)."""
    _slug(kind, "kind")
    _slug(name, "name")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt text is empty")
    root = _scope_root(scope, show)
    if scope == "project" and not root.parent.is_dir():
        raise ValueError(f"no show {project.active_project() if show is None else show!r}")
    folder = root / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    data = {"schema": SCHEMA, "kind": kind, "name": name, "prompt": prompt, "notes": notes,
            "updated": datetime.now().isoformat(timespec="seconds"), "from": origin}
    tmp = folder / f".{name}.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return _load(path, scope)  # type: ignore[return-value]
