"""The hub: one long-lived HTTP process the app and every Claude session share (Studio L4).

`python -m wzrd_mcp` serves `/mcp` (agents) and `/api/*` (the app's JSON)
from one Starlette app on 127.0.0.1:8787. This module owns the parts that
make it a *hub*:

- **Singleton** — `hub.lock` (`{pid, port, started}` in the library's
  `.wzrd/`). A second start finds a live hub answering `/api/hello` and
  exits 0; a stale lock is taken over.
- **Idle exit** — no request, no in-flight call and no running job for
  `idle_s` (default 1 h) → SIGTERM to itself (uvicorn shuts down cleanly).
  The long-lived `GET /mcp` stream a connected Claude session holds open
  does not count as activity.
- **`/api/*`** — thin JSON routes over `library.py` / `jobs.py`. Errors are
  `{error}` with 400 (bad request) or 500. Destructive operations (delete)
  exist only here, never as MCP tools (S13).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastmcp.exceptions import ToolError
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from . import jobs, library
from ._log import _emit
from .server import mcp

APP_NAME = "wzrd-hub"
API_VERSION = 1
IDLE_EXIT_S = 3600.0
_started = time.time()


def hello() -> dict:
    return {"app": APP_NAME, "version": mcp.version, "api": API_VERSION, "pid": os.getpid(),
            "media_root": str(library.media_root()), "db": str(library.db_path()),
            "schema": library.SCHEMA_VERSION, "started": _started}


# ---------------------------------------------------------------------------
# Singleton lock
# ---------------------------------------------------------------------------
def lock_path() -> Path:
    return library.wzrd_dir() / "hub.lock"


def _hub_answers(port: int) -> Optional[dict]:
    import httpx

    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/hello", timeout=2)
        data = r.json()
        return data if data.get("app") == APP_NAME else None
    except Exception:  # noqa: BLE001
        return None


def acquire_lock(port: int) -> Optional[dict]:
    """Take `hub.lock`. Returns the running hub's hello instead if one answers."""
    path = lock_path()
    try:
        held = json.loads(path.read_text())
    except (OSError, ValueError):
        held = None
    if held and held.get("pid") != os.getpid():
        running = _hub_answers(int(held.get("port", port)))
        if running:
            return running
    path.write_text(json.dumps({"pid": os.getpid(), "port": port, "started": _started}) + "\n")
    return None


def release_lock() -> None:
    path = lock_path()
    try:
        if json.loads(path.read_text()).get("pid") == os.getpid():
            path.unlink()
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Activity + idle exit
# ---------------------------------------------------------------------------
class Activity:
    """ASGI middleware: stamps the last request and counts in-flight calls."""

    last = time.time()
    inflight = 0

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # The streaming GET a connected MCP client holds open isn't activity.
        counted = not (scope.get("method") == "GET" and scope.get("path", "").startswith("/mcp"))
        if counted:
            Activity.last = time.time()
            Activity.inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            if counted:
                Activity.inflight -= 1
                Activity.last = time.time()


def start_idle_watch(idle_s: float) -> None:
    if idle_s <= 0:
        return

    def watch() -> None:
        while True:
            time.sleep(min(30.0, max(1.0, idle_s / 4)))
            idle = time.time() - Activity.last
            if Activity.inflight == 0 and jobs.running_here() == 0 and idle >= idle_s:
                _emit(f"[hub] idle for {idle:.0f}s with no running job — shutting down")
                release_lock()
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=watch, name="wzrd-hub-idle", daemon=True).start()


# ---------------------------------------------------------------------------
# /api routes
# ---------------------------------------------------------------------------
Handler = Callable[[Request, dict], Awaitable[Any]]


def api(path: str, methods: tuple[str, ...] = ("GET",)):
    def deco(fn: Handler) -> Handler:
        async def handler(request: Request) -> Response:
            try:
                body: dict = {}
                if request.method in ("POST", "PUT"):
                    raw = await request.body()
                    body = json.loads(raw) if raw.strip() else {}
                    if not isinstance(body, dict):
                        raise ValueError("body must be a JSON object")
                out = await fn(request, body)
                return out if isinstance(out, Response) else JSONResponse(out)
            except (ValueError, KeyError, ToolError, FileNotFoundError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception as exc:  # noqa: BLE001
                _emit(f"[hub] {request.method} {path} failed: {exc!r}")
                return JSONResponse({"error": str(exc)}, status_code=500)

        mcp.custom_route(path, methods=list(methods))(handler)
        return fn

    return deco


def _q(request: Request, key: str, default: Any = None) -> Any:
    return request.query_params.get(key, default)


def _thread(fn, *args, **kwargs):
    return asyncio.to_thread(fn, *args, **kwargs)


@api("/api/hello")
async def _hello(request, body):
    return hello()


# -- roots ------------------------------------------------------------------
@api("/api/roots", ("GET", "POST"))
async def _roots(request, body):
    if request.method == "POST":
        return await _thread(library.add_root, body["path"])
    return {"roots": library.roots()}


@api("/api/roots/remove", ("POST",))
async def _roots_remove(request, body):
    return await _thread(library.remove_root, body["path"])


# -- library ----------------------------------------------------------------
@api("/api/library/list")
async def _list(request, body):
    return await _thread(library.list_folder, _q(request, "dir") or str(library.media_root()))


@api("/api/library/item")
async def _item(request, body):
    ref = _q(request, "id") or _q(request, "path")
    if not ref:
        raise ValueError("pass ?id= or ?path=")
    return await _thread(library.render, ref)


@api("/api/library/index", ("POST",))
async def _index(request, body):
    out = await _thread(library.index_tree, body.get("path") or body["dir"], bool(body.get("deep", True)))
    return {**out, "pending": library.pending_count()}


@api("/api/library/label", ("POST",))
async def _label(request, body):
    refs = list(body.get("ids") or []) + list(body.get("paths") or [])
    if not refs:
        raise ValueError("pass ids and/or paths")
    kw: dict[str, Any] = {k: body[k] for k in ("tags", "add_tags", "remove_tags") if k in body}
    for k in ("score", "notes"):
        if k in body:
            kw[k] = body[k]
    return {"renders": await _thread(library.label, refs, **kw)}


@api("/api/library/move", ("POST",))
async def _move(request, body):
    dest = body.get("dest") or body["dest_dir"]
    return {"moved": await _thread(library.move, body["paths"], dest, bool(body.get("siblings", True)))}


@api("/api/library/rename", ("POST",))
async def _rename(request, body):
    return {"renamed": await _thread(library.rename, body["path"], body["name"], bool(body.get("siblings", True)))}


@api("/api/library/delete", ("POST",))
async def _delete(request, body):
    return await _thread(library.delete, body["paths"], bool(body.get("siblings", True)))


@api("/api/library/delete_scored", ("POST",))
async def _delete_scored(request, body):
    return await _thread(library.purge, body["dir"], int(body["max_score"]),
                         bool(body.get("recursive", False)), bool(body.get("dry_run", False)))


@api("/api/library/signal", ("POST",))
async def _signal(request, body):
    return await _thread(library.add_signal, body.get("id") or body["path"], body["kind"], body.get("seconds"))


@api("/api/library/tags")
async def _tags(request, body):
    return {"tags": library.all_tags()}


@api("/api/thumb")
async def _thumb(request, body):
    rid = _q(request, "id")
    if not rid:
        raise ValueError("pass ?id=<render id>")
    path = library.thumb_path(rid)
    if not path.is_file():
        return JSONResponse({"error": "no thumbnail"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


# -- selection (L5) ---------------------------------------------------------
@api("/api/selection", ("GET", "PUT", "POST"))
async def _selection(request, body):
    if request.method in ("PUT", "POST"):
        return library.set_selection(body)
    return library.selection()


# -- jobs + spend -----------------------------------------------------------
@api("/api/jobs")
async def _jobs(request, body):
    """Long-poll: `?version=<v>&wait=<s>` returns when the job set changed
    (or after `wait` s, max 30). `?since=<epoch>` keeps running jobs plus
    those started or ended since then."""
    seen = _q(request, "version")
    wait = min(float(_q(request, "wait", 0) or 0), 30.0)
    deadline = time.time() + wait
    while seen is not None and str(jobs.version()) == seen and time.time() < deadline:
        await asyncio.sleep(0.25)
    rows = await _thread(jobs.recent, int(_q(request, "limit", 50) or 50))
    since = _q(request, "since")
    if since is not None:
        t = float(since)
        rows = [j for j in rows if j["status"] == "running" or j["started"] >= t or (j["ended"] or 0) >= t]
    return {"version": jobs.version(), "now": time.time(), "jobs": rows, "spend": jobs.spend()}


@api("/api/jobs/cancel", ("POST",))
async def _jobs_cancel(request, body):
    return await jobs.cancel(body["job"])


@api("/api/estimate", ("POST",))
async def _estimate(request, body):
    tool = body["tool"]
    if tool not in jobs.ESTIMATORS:
        raise ValueError(f"no estimate for {tool!r} (free, or not a render tool)")
    return {"tool": tool, "usd": jobs.estimate(tool, body.get("args") or {}, int(body.get("count", 1))),
            "spend": jobs.spend()}


@api("/api/render", ("POST",))
async def _render(request, body):
    """Start `count` render jobs from the app (S16): each is the MCP tool
    called with `wait=false`, recorded `by: "app"`. Errors before submit
    (bad args, spend cap, unknown region) come back as 400."""
    name, args, count = body["tool"], dict(body.get("args") or {}), int(body.get("count", 1))
    t = await mcp.get_tool(name)
    if t is None or "render" not in (t.tags or set()):
        raise ValueError(f"{name!r} is not a render tool")
    if not 1 <= count <= 12:
        raise ValueError("count must be 1-12")
    token = jobs.current_by.set("app")
    started = []
    try:
        for _ in range(count):
            res = await t.run({**args, "wait": False})
            started.append(json.loads(res.content[0].text))
    finally:
        jobs.current_by.reset(token)
    return {"jobs": started}


@api("/api/jobs/item")
async def _job(request, body):
    return jobs.get(_q(request, "job") or "")


@api("/api/spend", ("GET", "POST"))
async def _spend(request, body):
    if request.method == "POST":
        jobs.set_daily_cap(float(body["cap_usd"]))
    return jobs.spend()


@api("/api/recipes")
async def _recipes(request, body):
    from .recipe_tools import recipes_table

    return await _thread(recipes_table, _q(request, "name"))
