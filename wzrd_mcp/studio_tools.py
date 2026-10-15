"""MCP tools for the Studio side of the hub: render jobs (H2), the selection,
the render database, agent labels and file moves (H3).

The app reaches the same state over `/api/*` (`hub.py`); these are the
agent's view. Agents may watch and cancel jobs; deleting media exists only
on `/api` (S13, S18).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import jobs as _jobs
from . import library
from .server import tool

_COMPACT = ("job", "tool", "status", "by", "project", "elapsed_s", "eta_s", "duration_s",
            "est_usd", "outputs", "error")


@tool(title="Render jobs", tags={"studio", "jobs"}, read_only=True, idempotent=True)
async def jobs(
    job: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10,
    ctx: Optional[Context] = None,
) -> dict:
    """Render jobs in the hub — yours, other chats' and the app's: status, elapsed / ETA, cost, outputs.

    A render started with wait=false returns a job id at once; poll here
    until it reads done (outputs listed) or failed. Statuses: running, done,
    failed, cancelled, lost (the process running it died mid-render).

    Args:
        job: One job id for its full row (args included). Omit for the newest jobs.
        status: Only jobs with this status (running, done, failed, cancelled, lost).
        limit: How many of the newest jobs (1-50).
    """
    if job:
        return _jobs.get(job)
    rows = _jobs.recent(max(1, min(limit, 50)), status)
    return {"jobs": [{k: j.get(k) for k in _COMPACT if j.get(k) not in (None, [])} for j in rows],
            "spend": _jobs.spend()}


@tool(title="Cancel render job", tags={"studio", "jobs"}, open_world=True)
async def cancel_job(job: str, ctx: Optional[Context] = None) -> dict:
    """Stop a running render: the remote GPU / API call is cancelled (TextureFlow on Modal, Kling on FAL) and the job reads cancelled, with no output.

    Nano Banana has no remote cancel: its request is dropped, but Google may
    still bill it. Only cancel what the operator asked to stop, or your own
    renders that went wrong.

    Args:
        job: The job id (from a wait=false reply or jobs).
    """
    return await _jobs.cancel(job)


# ---------------------------------------------------------------------------
# H3: the Studio selection, the render database, agent labels and file moves
# ---------------------------------------------------------------------------
_MEDIA_ARGS = {"images", "image_urls", "image_url", "start_image_url", "tail_image_url",
               "end_image_url", "control_input", "diffusion_mask", "video", "videos"}
_LOCAL_ARGS = {"region", "allow_over_cap", "hap", "project", "dest", "wait", "model", "backend"}


@tool(title="Studio selection", tags={"studio"}, read_only=True, idempotent=True)
async def studio_selection(ctx: Optional[Context] = None) -> dict:
    """What the operator has selected in the app's Studio room: the style tray, the selected items, the target folder and the open show.

    "These styles" / "these clips" / "put them here" mean this. Pass
    `styles` as a render's style images (texture_flow images=, nano_banana
    image_urls=); renders land in `target` by default (no dest= needed);
    pass `project` as project= on show calls.
    """
    sel = library.selection()
    target = library.target_dir()

    def describe(path: str) -> dict:
        row = library.query_one("SELECT render_id FROM files WHERE path=?", (path,))
        return {"path": path, "exists": Path(path).is_file(),
                **({"id": row["render_id"]} if row else {})}

    return {
        "styles": [describe(p) for p in sel.get("styles") or []],
        "items": [describe(p) for p in sel.get("items") or []],
        "target": str(target),
        "target_is_inbox": target == library.inbox(),
        "project": sel.get("project"),
    }


async def _defaults(tool_name: str) -> dict:
    """A tool's published argument defaults (to show only the args a render changed)."""
    from .server import mcp

    t = await mcp.get_tool(tool_name)
    props = ((t.parameters if t else {}) or {}).get("properties", {})
    return {k: v["default"] for k, v in props.items() if "default" in v}


def _style_names(record: dict) -> list[str]:
    images = (record.get("args") or {}).get("images") or (record.get("args") or {}).get("image_urls") or []
    by_path = {i.get("path"): i.get("name") for i in record.get("inputs") or []}
    return [by_path.get(i) or Path(str(i)).name for i in images]


def _best_file(render_id: str) -> tuple[Optional[str], Optional[str]]:
    """(master path, hap path) of a render's present files."""
    master = hap = None
    for r in library.query("SELECT path, variant FROM files WHERE render_id=? AND missing=0", (render_id,)):
        if not Path(r["path"]).is_file():
            continue
        if r["variant"] == "hap":
            hap = hap or r["path"]
        else:
            master = master or r["path"]
    return master, hap


@tool(title="Find renders", tags={"studio", "library"}, read_only=True, idempotent=True)
async def find_renders(
    tool: Optional[str] = None,
    min_score: Optional[int] = None,
    tag: Optional[str] = None,
    style: Optional[str] = None,
    recipe: Optional[str] = None,
    project: Optional[str] = None,
    text: Optional[str] = None,
    limit: int = 20,
    ctx: Optional[Context] = None,
) -> dict:
    """Search the render database — every WZRD render ever made, with the operator's 0-9 score, tags and notes — to learn what worked before a batch.

    Rows are best-scored first, then newest; `args` lists only what the
    render changed from the tool's defaults. Deleted renders keep their
    record (`purged`, no path) — still evidence. Unrated means unjudged,
    not bad. Before a batch: find_renders(style=…, tool=…) and reuse the
    settings of the ≥ 7s; avoid what scored ≤ 3.

    Args:
        tool: Only this tool's renders (texture_flow, nano_banana, kling_v25_image_to_video, …).
        min_score: Only renders the operator scored at least this (0-9).
        tag: Only renders carrying this tag (operator's or agent's).
        style: A style image — its path (matched by content, wherever it moved) or part of its file name.
        recipe: Only renders made with this style recipe.
        project: Only renders made for this show.
        text: Case-insensitive match in prompt/args, tags and notes.
        limit: Max rows (1-100).
    """
    where, params = ["kind = 'wzrd'"], []
    if tool:
        where.append("tool = ?")
        params.append(tool)
    if min_score is not None:
        where.append("score >= ?")
        params.append(int(min_score))
    if project:
        where.append("project = ?")
        params.append(project)
    rows = library.query(
        f"SELECT * FROM renders WHERE {' AND '.join(where)} "
        "ORDER BY score IS NULL, score DESC, created DESC", params)
    style_fp = None
    if style and Path(style).expanduser().is_file():
        from . import identity

        style_fp = identity.fingerprint(Path(style).expanduser())
    needle = (text or "").lower()
    defaults: dict[str, dict] = {}
    out = []
    for r in rows:
        rec = library._loads(r["record"], {})
        if rec.get("variant") == "hap" and rec.get("tool") == "to_hap":
            continue  # a HAP made from a foreign file: no creative record
        tags = library._loads(r["tags"], [])
        agent_tags = library._loads(r["agent_tags"], [])
        args = rec.get("args") or {}
        if tag and tag not in tags and tag not in agent_tags:
            continue
        if recipe and args.get("recipe") != recipe:
            continue
        if style:
            inputs = rec.get("inputs") or []
            if style_fp:
                if not any(i.get("fp") == style_fp for i in inputs):
                    continue
            elif not any(style.lower() in str(i.get("name", "")).lower() for i in inputs):
                continue
        if needle and needle not in " ".join(
                [json.dumps(args, default=str), r["notes"] or "", *tags, *agent_tags]).lower():
            continue
        if r["tool"] not in defaults:
            defaults[r["tool"]] = await _defaults(r["tool"]) if r["tool"] else {}
        d = defaults[r["tool"]]
        changed = {k: v for k, v in args.items()
                   if k not in _MEDIA_ARGS | _LOCAL_ARGS and d.get(k, object()) != v}
        master, hap = _best_file(r["id"])
        row = {"id": r["id"], "path": master or hap, "clip": hap if master else None,
               "tool": r["tool"], "score": r["score"], "tags": tags, "agent_tags": agent_tags,
               "notes": r["notes"], "args": changed, "styles": _style_names(rec),
               "project": r["project"], "region": r["region"], "created": r["created"],
               "purged": bool(r["purged"]) or not (master or hap)}
        out.append({k: v for k, v in row.items() if v not in (None, [], {}, False)})
        if len(out) >= max(1, min(limit, 100)):
            break
    return {"renders": out, "matched": len(out)}


@tool(title="Label render", tags={"studio", "library"}, idempotent=True)
async def label_render(
    render: str,
    tags: Optional[list[str]] = None,
    note: Optional[str] = None,
    remove_tags: Optional[list[str]] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Tag a render or add a note — stored as the agent's tags, beside (never over) the operator's. The 0-9 score is the operator's alone.

    Tag what you can see or know: content ("jellyfish", "embers"), motion
    ("slow pulse"), mood, what it was made for. Notes are appended, signed
    "[agent]".

    Args:
        render: Its file path (mp4, its HAP .mov, image) or render id (r_…).
        tags: Agent tags to add.
        note: A line appended to the render's notes.
        remove_tags: Agent tags to remove.
    """
    try:
        rid = library.resolve_id(render)
        if note:
            row = library.query_one("SELECT notes FROM renders WHERE id=?", (rid,))
            current = (row["notes"] or "").rstrip()
            library.label([rid], notes=f"{current}\n[agent] {note}".strip(), agent=True)
        [out] = library.label([rid], add_tags=tags, remove_tags=remove_tags, agent=True)
    except ValueError as exc:
        raise ToolError(str(exc))
    return {k: out[k] for k in ("id", "score", "tags", "agent_tags", "notes")}


def _paths_of(render: str) -> list[str]:
    """A path as given, or every present file of a render id (master + HAP)."""
    if render.startswith(("r_", "fp_")):
        rows = library.query("SELECT path FROM files WHERE render_id=? AND missing=0", (render,))
        paths = [r["path"] for r in rows if Path(r["path"]).is_file()]
        if not paths:
            raise ToolError(f"render {render} has no file left on disk")
        return paths
    return [render]


@tool(title="Move render", tags={"studio", "library"})
async def move_render(render: str, folder: str, ctx: Optional[Context] = None) -> dict:
    """Move a render's files (the mp4 and its HAP travel together) into another folder; labels and record follow it. Never deletes.

    Args:
        render: Its file path or render id.
        folder: Absolute destination folder (created if missing), e.g. a subfolder of the media library.
    """
    if not Path(folder).expanduser().is_absolute():
        raise ToolError("folder must be an absolute path")
    try:
        moved = await asyncio.to_thread(library.move, _paths_of(render), folder, True)
    except ValueError as exc:
        raise ToolError(str(exc))
    return {"moved": moved}


@tool(title="Rename render", tags={"studio", "library"})
async def rename_render(render: str, name: str, ctx: Optional[Context] = None) -> dict:
    """Rename a render's files (new stem, extensions kept; its HAP is renamed with it). Labels and record follow.

    Args:
        render: Its file path or render id.
        name: The new file name without extension, e.g. "ember-veil-tower-01".
    """
    try:
        renamed = await asyncio.to_thread(library.rename, _paths_of(render)[0], name, True)
    except ValueError as exc:
        raise ToolError(str(exc))
    return {"renamed": renamed}
