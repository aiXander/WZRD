"""MCP tools for the Studio side of the hub: render jobs (H2).

The app reaches the same state over `/api/*` (`hub.py`); these are the
agent's view. Agents may watch and cancel jobs; deleting media exists only
on `/api` (S13, S18).
"""

from __future__ import annotations

from typing import Optional

from fastmcp import Context

from . import jobs as _jobs
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
