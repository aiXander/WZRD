"""Provenance sidecars, the per-project spend cap, and `region=` lookup.

Gig contract C2: every media file a tool writes into the project gets a
`<file>.meta.json` beside it — which tool made it, from what, at what cost,
for which region. The app's clip grid files clips into region rows from it;
the spend cap sums its `est_usd` across the project.

Tools never write sidecars themselves. `logged_tool` calls `write_sidecars`
after every successful call: outputs are the media paths in the result that
the call created inside the project. A tool that knows more (cost, region,
the model it ran) returns a private `_provenance` dict, popped before the
result reaches the agent:

    {"est_usd": {path: usd}, "args": {...}, "region": id, "source_box": [...],
     "parent": path | {output path: path}}

`est_usd` is keyed per output so a multi-output call (TextureFlow's preview
videos) is counted once, on the file that cost the money.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastmcp.exceptions import ToolError

from . import project

SIDECAR_SUFFIX = ".meta.json"
THUMB_SUFFIX = ".thumb.jpg"  # a clip's preview (C3) — derived, no sidecar of its own
SIDECAR_SCHEMA = 1
WARN_AT = 0.8
_MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".webm"}


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------
def sidecar_path(path: str | Path) -> Path:
    return Path(f"{path}{SIDECAR_SUFFIX}")


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _inputs(client_args: dict) -> list[str]:
    """Local files (resolved) and URLs among the call's arguments, in order."""
    seen: list[str] = []
    for s in _strings(client_args):
        if s.startswith(("http://", "https://")):
            ref = s
        else:
            p = Path(s).expanduser()
            if not (p.is_absolute() or os.sep in s) or not p.is_file():
                continue
            ref = str(p.resolve())
        if ref not in seen:
            seen.append(ref)
    return seen


def _outputs(result: dict, since: float, inputs: list[str]) -> list[str]:
    """Media files in *result* that this call wrote inside the active project."""
    root = project.project_dir().resolve()
    found: list[str] = []
    for s in _strings(result):
        if not s.startswith("/") or s.endswith((SIDECAR_SUFFIX, THUMB_SUFFIX)):
            continue
        p = Path(s)
        if p.suffix.lower() not in _MEDIA_EXTS or not p.is_file():
            continue
        p = p.resolve()
        if str(p) in inputs or str(p) in found or not p.is_relative_to(root):
            continue
        if p.stat().st_mtime < since - 2:
            continue
        found.append(str(p))
    return found


def write_sidecars(tool: str, client_args: dict, result: dict, since: float,
                   prov: Optional[dict] = None) -> list[str]:
    """Write a C2 sidecar next to every output of one tool call. Returns them."""
    prov = prov or {}
    inputs = _inputs(client_args)
    outputs = _outputs(result, since, inputs)
    est = prov.get("est_usd") or {}
    local_inputs = [i for i in inputs if i.startswith("/")]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written = []
    default_parent = local_inputs[0] if local_inputs else None
    parents = prov.get("parent", default_parent)
    for out in outputs:
        meta = {
            "schema": SIDECAR_SCHEMA,
            "tool": tool,
            "args": {k: v for k, v in {**client_args, **(prov.get("args") or {})}.items() if v is not None},
            "inputs": inputs,
            "parent": parents.get(out, default_parent) if isinstance(parents, dict) else parents,
            "created": created,
            "est_usd": round(float(est.get(out, 0.0)), 4),
            "region": prov.get("region"),
            "source_box": prov.get("source_box"),
        }
        path = sidecar_path(out)
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
        written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# Spend cap (D10): soft, per project, summed from the sidecars
# ---------------------------------------------------------------------------
def spend_cap() -> float:
    meta = project.read_project_file() or {}
    return float(meta.get("spend_cap_usd", project.DEFAULT_SPEND_CAP_USD))


def spent() -> float:
    """Sum of `est_usd` over every sidecar in the active project."""
    root = project.project_dir()
    if not root.is_dir():
        return 0.0
    total = 0.0
    for p in root.rglob(f"*{SIDECAR_SUFFIX}"):
        try:
            total += float(json.loads(p.read_text()).get("est_usd") or 0.0)
        except (ValueError, OSError):
            continue
    return total


def check_spend(est_usd: float, allow_over_cap: bool) -> None:
    """Refuse a paid call that would take the project past its cap."""
    total, cap = spent(), spend_cap()
    if total + est_usd > cap and not allow_over_cap:
        raise ToolError(
            f"spend cap: project {project.active_project()!r} has spent ${total:.2f} of its "
            f"${cap:.2f} cap and this call costs ~${est_usd:.2f}. Ask Xander first; then call "
            f"again with allow_over_cap=true, or raise the cap with set_project(spend_cap_usd=…)."
        )


def spend_report() -> dict:
    """The `spend` line appended to every paid tool's result (after its sidecars)."""
    total, cap = spent(), spend_cap()
    out: dict[str, Any] = {"spend": f"${total:.2f} of ${cap:.2f} cap"}
    if total >= WARN_AT * cap:
        out["spend_warning"] = (
            f"project has used {total / cap:.0%} of its spend cap — tell Xander before "
            f"spending more" if total < cap else
            "project is over its spend cap — every further paid call needs Xander's OK"
        )
    return out


# ---------------------------------------------------------------------------
# region= : a pack layer id -> its placement box
# ---------------------------------------------------------------------------
def _pack_json() -> Path:
    root = project.project_dir()
    scene = root / "scene.json"
    if scene.is_file():
        try:
            pack = json.loads(scene.read_text()).get("pack")
        except ValueError:
            pack = None
        if pack:
            path = (root / pack / "pack.json").resolve()
            if path.is_file():
                return path
    packs = sorted((root / project.PACKS).glob("*/pack.json"), key=lambda p: p.stat().st_mtime)
    if not packs:
        raise ToolError(
            f"region=: project {project.active_project()!r} has no layer pack yet — "
            f"run build_layerpack first (regions are pack layer ids)"
        )
    return packs[-1]


def region_meta(layer_id: str) -> dict:
    """`{"region": id, "source_box": [x0, y0, x1, y1]}` for a pack layer.

    The box is the layer's padded `source_box` (C4), else its `bbox` — the
    same fallback the engine uses to place a clip in that region.
    """
    pack = _pack_json()
    layers = json.loads(pack.read_text()).get("layers", [])
    for layer in layers:
        if layer.get("id") == layer_id:
            return {"region": layer_id, "source_box": layer.get("source_box") or layer.get("bbox")}
    ids = ", ".join(sorted(str(layer.get("id")) for layer in layers))
    raise ToolError(f"region={layer_id!r} is not a layer of {pack.parent.name} (layers: {ids})")

