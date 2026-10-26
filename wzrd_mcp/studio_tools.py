"""MCP tools for the Studio side of the hub: render jobs (H2) and the
operator's selection (H3).

The app reaches the same state over `/api/*` (`hub.py`); these are the
agent's read-only view. Labels, scores, housekeeping (cancel, move, rename,
delete) and deck curation are the operator's, in the app only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastmcp import Context

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


# ---------------------------------------------------------------------------
# H3: the Studio selection
# ---------------------------------------------------------------------------
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
