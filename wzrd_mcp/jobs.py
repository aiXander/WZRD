"""Render jobs and the daily spend cap (Studio S15, S17; `jobs` table, L3).

Every call of a tool registered with `render=True` is one job row: started
by `logged_tool`, finished with its outputs and recorded cost. A job whose
process died before `ended` was written reads **lost**. The spend cap is
daily and global: today's (local midnight on) `est_usd` over all jobs —
running ones count with their pre-estimate, so a parallel batch can't
overshoot.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional

from fastmcp.exceptions import ToolError

from . import identity, library

DEFAULT_DAILY_CAP_USD = 20.0
WARN_AT = 0.8
ETA_SAMPLES = 20

#: The job of the tool call running in this context (None outside a render call).
current_job: ContextVar[Optional[str]] = ContextVar("wzrd_current_job", default=None)
#: Who started the call: "agent" (an MCP tool call) or "app" (`/api/render`).
current_by: ContextVar[str] = ContextVar("wzrd_job_by", default="agent")

#: Size args per tool: jobs with the same values share an ETA.
SIZE_ARGS = {
    "texture_flow": ("width", "height", "n_seconds", "use_upscale", "upscale_esrgan", "n_steps", "recipe"),
    "nano_banana": ("resolution", "num_images", "thinking_level"),
    "kling_v25_image_to_video": ("duration",),
    "kling_v3_image_to_video": ("duration", "generate_audio"),
    "to_hap": ("codec",),
}

# Bumped on every job change; `/api/jobs` long-polls on it.
_version = 0
_version_lock = threading.Lock()


def _bump() -> None:
    global _version
    with _version_lock:
        _version += 1


def version() -> int:
    return _version


def size_key(tool: str, args: dict) -> str:
    return json.dumps({k: args.get(k) for k in SIZE_ARGS.get(tool, ())}, sort_keys=True)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def start(tool: str, args: dict, *, by: str = "agent", session: Optional[str] = None,
          project: Optional[str] = None, dest: Optional[str] = None) -> str:
    job = identity.new_id("j_")
    with library.tx() as c:
        c.execute(
            "INSERT INTO jobs(id, tool, args, by, session, pid, started, dest, size_key, project) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job, tool, library._dumps(args), by, session, os.getpid(), time.time(), dest,
             size_key(tool, args), project),
        )
    _bump()
    return job


def update(job: str, **fields: Any) -> None:
    allowed = {"est_usd", "remote", "dest"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    with library.tx() as c:
        c.execute(f"UPDATE jobs SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?", (*sets.values(), job))
    _bump()


def finish(job: str, *, ok: bool, error: Optional[str] = None, outputs: Optional[list[str]] = None,
           est_usd: Optional[float] = None, cancelled: bool = False) -> None:
    now = time.time()
    with library.tx() as c:
        row = c.execute("SELECT started, est_usd FROM jobs WHERE id=?", (job,)).fetchone()
        if not row:
            return
        c.execute(
            "UPDATE jobs SET ended=?, ok=?, error=?, outputs=?, est_usd=?, duration_s=?, cancelled=? WHERE id=?",
            (now, int(ok), error, library._dumps(outputs or []),
             round(float(est_usd if est_usd is not None else row["est_usd"]), 4),
             round(now - row["started"], 2), int(cancelled), job),
        )
    _bump()


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status_of(row: Any) -> str:
    if row["ended"] is None:
        return "running" if _alive(row["pid"]) else "lost"
    if row["cancelled"]:
        return "cancelled"
    return "done" if row["ok"] else "failed"


def _job_dict(row: Any) -> dict:
    st = status_of(row)
    out = {
        "job": row["id"], "tool": row["tool"], "status": st, "by": row["by"],
        "args": library._loads(row["args"], {}), "started": row["started"], "ended": row["ended"],
        "duration_s": row["duration_s"], "est_usd": row["est_usd"], "error": row["error"],
        "outputs": library._loads(row["outputs"], []), "dest": row["dest"], "project": row["project"],
        "session": row["session"],
    }
    if st == "running":
        out["elapsed_s"] = round(time.time() - row["started"], 1)
        e = eta(row["tool"], row["size_key"])
        out["eta_s"] = round(max(0.0, e - out["elapsed_s"]), 1) if e is not None else None
    return out


def get(job: str) -> dict:
    row = library.query_one("SELECT * FROM jobs WHERE id=?", (job,))
    if not row:
        raise ToolError(f"no job {job!r}")
    return _job_dict(row)


def recent(limit: int = 50, status: Optional[str] = None) -> list[dict]:
    rows = library.query("SELECT * FROM jobs ORDER BY started DESC LIMIT ?", (max(1, min(limit, 500)),))
    jobs = [_job_dict(r) for r in rows]
    return [j for j in jobs if j["status"] == status] if status else jobs


def eta(tool: str, key: Optional[str]) -> Optional[float]:
    """Median `duration_s` of the last 20 successful jobs with the same tool + size args."""
    rows = library.query(
        "SELECT duration_s FROM jobs WHERE tool=? AND size_key IS ? AND ok=1 AND cancelled=0 "
        "AND duration_s IS NOT NULL ORDER BY started DESC LIMIT ?", (tool, key, ETA_SAMPLES))
    return round(statistics.median(r["duration_s"] for r in rows), 1) if rows else None


# ---------------------------------------------------------------------------
# Daily spend cap (S15)
# ---------------------------------------------------------------------------
def today_start() -> float:
    now = datetime.now().astimezone()
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def daily_cap() -> float:
    return float(library.get_setting("daily_cap_usd", DEFAULT_DAILY_CAP_USD))


def set_daily_cap(usd: float) -> None:
    if usd < 0:
        raise ValueError("the daily cap must be >= 0")
    library.set_setting("daily_cap_usd", float(usd))


def spent_today(exclude: Optional[str] = None) -> float:
    row = library.query_one("SELECT COALESCE(SUM(est_usd), 0) AS s FROM jobs WHERE started>=? AND id IS NOT ?",
                            (today_start(), exclude))
    return float(row["s"])


def running_here() -> int:
    """Jobs this process is running right now (the hub's idle check)."""
    row = library.query_one("SELECT COUNT(*) AS n FROM jobs WHERE ended IS NULL AND pid=?", (os.getpid(),))
    return int(row["n"])


def running_usd() -> float:
    rows = library.query("SELECT pid, ended, est_usd FROM jobs WHERE started>=? AND ended IS NULL", (today_start(),))
    return float(sum(r["est_usd"] for r in rows if _alive(r["pid"])))


def check_spend(est_usd: float, allow_over_cap: bool) -> None:
    """Refuse a paid call that would take today past the daily cap; record
    the pre-estimate on the running job so parallel calls see it."""
    job = current_job.get()
    total, cap = spent_today(exclude=job), daily_cap()
    if total + est_usd > cap and not allow_over_cap:
        raise ToolError(
            f"daily spend cap: ${total:.2f} of today's ${cap:.2f} is spent and this call costs "
            f"~${est_usd:.2f}. Ask Xander first; then call again with allow_over_cap=true, or he "
            f"raises the cap (Studio, or set_project(daily_cap_usd=…) on his word)."
        )
    if job:
        update(job, est_usd=round(float(est_usd), 4))


def spend() -> dict:
    total, cap = spent_today(), daily_cap()
    return {"today_usd": round(total, 4), "cap_usd": cap, "running_usd": round(running_usd(), 4),
            "warn_at": WARN_AT, "warn": total >= WARN_AT * cap, "over": total >= cap}


def spend_report() -> dict:
    """The `spend` line appended to every paid tool's result."""
    total, cap = spent_today(), daily_cap()
    out: dict[str, Any] = {"spend": f"${total:.2f} of ${cap:.2f} today"}
    if total >= WARN_AT * cap:
        out["spend_warning"] = (
            f"{total / cap:.0%} of today's spend cap is used — tell Xander before spending more"
            if total < cap else
            "today is over the spend cap — every further paid call needs Xander's OK"
        )
    return out
