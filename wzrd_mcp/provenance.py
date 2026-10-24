"""Render records, the daily spend cap, and `region=` lookup.

Studio contract L2 (replacing the gig's C2 `.meta.json` sidecars): every media
file a tool call creates gets a **render id** and a creation record, embedded
in the file itself as the `wzrd` tag (`identity.write_tag`) and stored as a
`renders` + `files` row in `library.db`. Labels live in the database, keyed by
the id, so renaming or moving the file never loses them.

Tools never write records themselves. `logged_tool` calls `record_outputs`
after every successful call: outputs are the media paths in the result that
the call created (anywhere — the library, a project, beside a source). A tool
that knows more returns a private `_provenance` dict, popped before the result
reaches the agent:

    {"est_usd": {path: usd}, "args": {...}, "region": id, "source_box": [...],
     "parent": path | render id | {output path: path}, "variant_of": {hap path: source path},
     "paid": bool, "record": {extra top-level record fields}}

`record` adds tool-specific fields to every master record of the call
(e.g. `rectify_surface`'s `corners` / `manual_corners`); the standard keys
always win.

`est_usd` is keyed per output so a multi-output call is counted once, on the
file that cost the money. `variant_of` marks a HAP made from a source in the
same call (or by `to_hap`): it shares the source's render id (S8).

Old sidecars are still read (`legacy_sidecar`) wherever a record is looked
up, and absorbed into the database when their folder is indexed.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastmcp.exceptions import ToolError

from . import identity, jobs, library, project

SIDECAR_SUFFIX = ".meta.json"  # legacy (gig C2), read-only now
THUMB_SUFFIX = ".thumb.jpg"  # legacy clip preview beside a .mov (gig C3), read-only now
RECORD_SCHEMA = identity.TAG_SCHEMA
_NO_RECORD = {"save_recipe"}  # writes into library/recipes — not renders


# ---------------------------------------------------------------------------
# Records (L2)
# ---------------------------------------------------------------------------
def sidecar_path(path: str | Path) -> Path:
    return Path(f"{path}{SIDECAR_SUFFIX}")


def legacy_sidecar(path: str | Path) -> Optional[dict]:
    try:
        return json.loads(sidecar_path(Path(path).resolve()).read_text())
    except (OSError, ValueError):
        return None


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
    """Media files in *result* that this call wrote (not inputs, not scratch)."""
    scratch = library.wzrd_dir()
    found: list[str] = []
    for s in _strings(result):
        if not s.startswith("/") or s.endswith((SIDECAR_SUFFIX, THUMB_SUFFIX)):
            continue
        p = Path(s)
        if identity.media_kind(p) is None or not p.is_file():
            continue
        p = p.resolve()
        if str(p) in inputs or str(p) in found or p.is_relative_to(scratch):
            continue
        if p.stat().st_mtime < since - 2:
            continue
        found.append(str(p))
    return found


def _input_ref(ref: str) -> dict:
    if ref.startswith(("http://", "https://")):
        return {"name": ref, "fp": None}
    try:
        fp = identity.fingerprint(ref)
    except OSError:
        fp = None
    return {"name": Path(ref).name, "path": ref, "fp": fp}


def _render_id_of(path: str) -> Optional[str]:
    try:
        return library.resolve_id(path)
    except Exception:  # noqa: BLE001
        return None


def _parent_id(parent: Optional[str]) -> Optional[str]:
    """A parent given as a render id (`r_…`, e.g. a finished draft whose mp4 may be gone) or a file path."""
    if not parent:
        return None
    if parent.startswith("r_") and library.query_one("SELECT 1 FROM renders WHERE id=?", (parent,)):
        return parent
    return _render_id_of(parent) if Path(parent).is_file() else None


def record_outputs(tool: str, client_args: dict, result: dict, since: float,
                   prov: Optional[dict] = None, job: Optional[str] = None) -> list[dict]:
    """Tag + record every output of one tool call. Returns `[{path, id}]`."""
    if tool in _NO_RECORD:
        return []
    prov = prov or {}
    inputs = _inputs(client_args)
    outputs = _outputs(result, since, inputs)
    if not outputs:
        return []
    est = prov.get("est_usd") or {}
    variant_of = {str(Path(k).resolve()): str(Path(v).resolve()) for k, v in (prov.get("variant_of") or {}).items()}
    local_inputs = [i for i in inputs if i.startswith("/")]
    default_parent = local_inputs[0] if local_inputs else None
    parents = prov.get("parent", default_parent)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    duration = round(time.time() - since, 2)
    canvas = project.canvas()
    input_refs = [_input_ref(i) for i in inputs]
    args = {k: v for k, v in {**client_args, **(prov.get("args") or {})}.items() if v is not None}
    ids: dict[str, str] = {}
    done = []
    # Masters first, so a HAP made in the same call finds its source's id.
    for out in sorted(outputs, key=lambda o: o in variant_of):
        if out in variant_of:
            src = variant_of[out]
            rid = ids.get(src) or _render_id_of(src) or f"fp_{identity.fingerprint(src)}"
            master = library.render(rid)["record"] if library.query_one(
                "SELECT 1 FROM renders WHERE id=?", (rid,)) else {}
            if master and prov.get("region") and not master.get("region"):
                # to_hap(region=) files a source that records none (a stitched story).
                master = {**master, "region": prov["region"], "source_box": prov.get("source_box")}
                with library.tx() as c:
                    c.execute("UPDATE renders SET region=?, record=? WHERE id=?",
                              (prov["region"], library._dumps(master), rid))
            if not master and prov.get("region"):
                with library.tx() as c:
                    c.execute("UPDATE renders SET region=? WHERE id=? AND region IS NULL", (prov["region"], rid))
            record = {**master, "id": rid, "variant": "hap"} if master else {
                "schema": RECORD_SCHEMA, "id": rid, "variant": "hap", "tool": tool, "args": args,
                "inputs": [_input_ref(src)], "parent": None, "created": created, "est_usd": 0.0,
                "duration_s": duration, "job": job, "project": project.active_project(), "canvas": None,
                "region": prov.get("region"), "source_box": prov.get("source_box"),
            }
        else:
            parent = parents.get(out, default_parent) if isinstance(parents, dict) else parents
            record = {
                **(prov.get("record") or {}),
                "schema": RECORD_SCHEMA, "id": identity.new_id(), "variant": "master", "tool": tool,
                "args": args, "inputs": input_refs,
                "parent": _parent_id(parent),
                "created": created, "est_usd": round(float(est.get(out, 0.0)), 4),
                "duration_s": duration, "job": job, "project": project.active_project(),
                "canvas": list(canvas) if canvas else None,
                "region": prov.get("region"), "source_box": prov.get("source_box"),
            }
        try:
            identity.write_tag(out, record)
        except Exception as exc:  # noqa: BLE001 — the database row still identifies it (by fp)
            from ._log import log_progress

            log_progress(tool, f"wzrd tag not embedded in {out}: {exc}")
        library.index_file(out, record=record)
        ids[out] = record["id"]
        done.append({"path": out, "id": record["id"]})
    return done


# ---------------------------------------------------------------------------
# Spend cap (S15): daily, global, summed from the jobs table — see jobs.py
# ---------------------------------------------------------------------------
def check_spend(est_usd: float, allow_over_cap: bool) -> None:
    jobs.check_spend(est_usd, allow_over_cap)


def spend_report() -> dict:
    return jobs.spend_report()


# ---------------------------------------------------------------------------
# region= : a pack layer id -> its placement box
# ---------------------------------------------------------------------------
def _pack_json(name: Optional[str] = None) -> Path:
    root = project.project_dir(name)
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
            f"region=: project {name or project.active_project()!r} has no layer pack yet — "
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


# Shape images for a region's render, best first: the named assets (canvas-sized
# by contract), then the pack's own surface.
REGION_SHAPE_ASSETS = ("surface_lit", "surface")


def region_inputs(layer_id: str, name: Optional[str] = None) -> dict:
    """A region's TextureFlow inputs, cut to its placement box.

    `mask` = the layer's mask (white = the region), `shape` = the show's
    surface (`surface_lit`, else `surface`, else the pack's surface image)
    blacked out outside the region, both cropped to the padded `source_box` —
    the box the engine fits the clip into, so a render at that aspect lands
    back on the region exactly. Outside the canvas the crop is black.

    Why black outside: TextureFlow keeps the shape input wherever the mask is
    black, so an unmasked surface would end up baked into the video around
    the region (the engine hides it at playback; the file shouldn't carry it).
    Cached in `<show>/regions/auto/`, re-cut when the pack or the source
    image is newer.
    """
    from PIL import Image

    from . import assets

    pack = _pack_json(name)
    manifest = json.loads(pack.read_text())
    layer = next((lay for lay in manifest.get("layers", []) if lay.get("id") == layer_id), None)
    if layer is None:
        ids = ", ".join(sorted(str(lay.get("id")) for lay in manifest.get("layers", [])))
        raise ToolError(f"region={layer_id!r} is not a layer of {pack.parent.name} (layers: {ids})")
    box = layer.get("source_box") or layer.get("bbox")
    canvas = manifest.get("projector_resolution")
    if not box or not canvas:
        raise ToolError(f"region={layer_id!r}: the pack has no box or canvas size for it")

    shape_src, shape_from = None, None
    for key in REGION_SHAPE_ASSETS:
        try:
            shape_src, shape_from = assets.resolve(key, name or project.active_project()), key
            break
        except ValueError:
            continue
    if shape_src is None and manifest.get("surface") and (pack.parent / manifest["surface"]).is_file():
        shape_src, shape_from = str(pack.parent / manifest["surface"]), "pack surface"

    out_dir = project.project_dir(name) / project.REGIONS / "auto"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pack.parent.name}_{layer_id}"

    def crop(src: Path, mode: str) -> Image.Image:
        with Image.open(src) as im:
            im = im.convert(mode)
            sx, sy = im.width / canvas[0], im.height / canvas[1]
            x0, y0, x1, y1 = box
            return im.crop((round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)))

    def stale(dest: Path, *srcs: Path) -> bool:
        return not dest.is_file() or dest.stat().st_mtime < max(p.stat().st_mtime for p in (pack, *srcs))

    mask_src = pack.parent / layer["mask"]
    mask_out = out_dir / f"{stem}_mask.png"
    if stale(mask_out, mask_src):
        crop(mask_src, "L").save(mask_out)
    shape_out = None
    if shape_src:
        shape_out = out_dir / f"{stem}_surface.png"
        if stale(shape_out, mask_src, Path(shape_src)):
            surface = crop(Path(shape_src), "RGB")
            mask = Image.open(mask_out).convert("L").resize(surface.size)
            Image.composite(surface, Image.new("RGB", surface.size), mask).save(shape_out)

    return {
        "region": layer_id,
        "source_box": box,
        "mask": str(mask_out),
        "shape": str(shape_out) if shape_out else None,
        "shape_from": shape_from,
    }

