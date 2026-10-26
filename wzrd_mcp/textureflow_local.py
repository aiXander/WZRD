"""Client for the local TextureFlow job server — `texture_flow(backend="local")`.

The server runs on Xander's desktop GPU (RTX 3090, over Tailscale) and lives in
the textureflow-lab repo (its job-server spec / `tflab/server.py`). This module
speaks its wire contract v1 (frozen): bearer-token JSON over HTTP,

    GET  /v1/health                    → {ok, gpu, busy, queue, worker, commit, vram_free_gb}
                                         (worker ready|booting|idle|waiting_gpu|down; idle/waiting_gpu = up)
    POST /v1/render  (multipart)       → {job_id, queue_position}; 400 {error} = validation
                                         (+ optional part `meta` {client, label}; an older server 400s
                                         on it → resent without, remembered)
    GET  /v1/queue                     → {running, queued: [...], worker, gpu_holder} (additive; 404 = older)
    POST /v1/queue/order {order}       → rewrites the server's dispatch order (additive)
    GET  /v1/jobs/{id}?wait=<s>        → {state, stage, step, total, queue_position, error} (long-poll)
    GET  /v1/jobs/{id}/result          → the Modal `render` dict, outputs as {filename, mime, url}
    GET  /v1/jobs/{id}/files/{name}    → the file bytes
    POST /v1/jobs/{id}/cancel          → {job_id, state}

The server owns the local queue (render-queue plan Q1): `tf_queue` only holds a job while the server is
unreachable. `render()` = submit → `collect()`; a hub restart re-adopts a submitted job with `collect()` alone.
Status polls run every ≈ 1 s (`wait=1`: the long-poll returns early only on state/stage changes, and the
step ticks are the progress bar) and feed `tf_queue.local_status`.

`render()` returns **exactly the dict Modal's `TextureFlow.render` returns**
(each output's `url` downloaded into `bytes`), so `tools.texture_flow`
post-processes both backends with one code path.

The backend is always the caller's explicit choice: nothing here ever falls
back to Modal. A desktop that is off fails the health check in ≈ 2 s.
Config: `TEXTUREFLOW_LOCAL_URL`, `TEXTUREFLOW_LOCAL_TOKEN`.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import httpx
from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import jobs
from ._keepalive import with_keepalive
from ._log import log_progress

HEALTH_TIMEOUT_S = 2.0
POLL_WAIT_S = 1  # long-poll window per status request (contract 0-60): ≈ 1 s so step ticks reach the bar
LOST_AFTER_S = 60.0  # a mid-job outage longer than this fails the call
_TERMINAL = ("done", "error", "cancelled")


def config() -> tuple[str, str]:
    """`(url, token)` from the env, or a ToolError naming what is missing."""
    url = (os.getenv("TEXTUREFLOW_LOCAL_URL") or "").strip().rstrip("/")
    token = (os.getenv("TEXTUREFLOW_LOCAL_TOKEN") or "").strip()
    if not url:
        raise ToolError("backend='local' needs TEXTUREFLOW_LOCAL_URL in WZRD's env (the desktop "
                        "job server, e.g. http://<desktop>.<tailnet>.ts.net:8790)")
    if not token:
        raise ToolError("backend='local' needs TEXTUREFLOW_LOCAL_TOKEN in WZRD's env (the job "
                        "server's bearer token)")
    return url, token


def _client(url: str, token: str, timeout: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {token}"},
                             timeout=timeout)


def _error_text(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error") or r.text)
    except ValueError:
        return r.text or f"HTTP {r.status_code}"


def _check(r: httpx.Response, url: str, what: str) -> None:
    if r.status_code == 401:
        raise ToolError(f"local TextureFlow at {url} refused the token (401) — check TEXTUREFLOW_LOCAL_TOKEN")
    if r.status_code >= 400:
        raise ToolError(f"local TextureFlow at {url}: {what} failed (HTTP {r.status_code}): {_error_text(r)}")


async def health() -> dict:
    """`GET /v1/health` within ≈ 2 s, or a ToolError saying why local can't render now."""
    url, token = config()
    try:
        async with _client(url, token, HEALTH_TIMEOUT_S) as c:
            r = await asyncio.wait_for(c.get("/v1/health"), HEALTH_TIMEOUT_S)
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        raise ToolError(f"local TextureFlow at {url} unreachable — is the desktop on? "
                        f"({type(exc).__name__}; nothing was sent to Modal)") from None
    _check(r, url, "health check")
    info = r.json()
    if not info.get("ok"):
        raise ToolError(f"local TextureFlow at {url} answered but isn't ok: {info}")
    # worker "booting" = up, the job just queues; "down" = the server still
    # accepts the job and fails it at once, which the first status poll reports.
    return info


async def probe() -> tuple[Optional[dict], Optional[str]]:
    """`health()` that never raises: `(info, None)` or `(None, why)`."""
    try:
        return await health(), None
    except ToolError as exc:
        return None, str(exc)


async def queue() -> Optional[dict]:
    """`GET /v1/queue` (additive route), or None: an older server (404) or unreachable."""
    try:
        url, token = config()
        async with _client(url, token, HEALTH_TIMEOUT_S + 1) as c:
            r = await c.get("/v1/queue")
    except (ToolError, httpx.HTTPError):
        return None
    if r.status_code != 200:
        return None
    try:
        out = r.json()
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


async def queue_order(order: list[str]) -> dict:
    """`POST /v1/queue/order {order}` — the server's queued jobs in a new dispatch order."""
    url, token = config()
    async with _client(url, token, 10.0) as c:
        r = await c.post("/v1/queue/order", json={"order": order})
    if r.status_code == 404:
        raise ToolError("the desktop job server has no /v1/queue/order (an older version) — it can't reorder yet")
    _check(r, url, "reorder")
    return r.json()


async def cancel(job_id: str) -> dict:
    """`POST /v1/jobs/{id}/cancel` — also what `jobs.cancel` runs for a `local|<id>` remote.

    A running job answers `"state": "running"` until the worker stops, then
    always ends `cancelled`; `_wait` keeps polling until a terminal state."""
    url, token = config()
    async with _client(url, token, 10.0) as c:
        r = await c.post(f"/v1/jobs/{job_id}/cancel")
    _check(r, url, f"cancel of job {job_id}")
    return r.json()


def _describe(st: dict) -> str:
    if st["state"] == "queued":
        return f"queued (position {st.get('queue_position')})"
    stage = st.get("stage") or st["state"]
    if st.get("step") is not None and st.get("total"):
        return f"{stage} {st['step']}/{st['total']}"
    return stage


class _Hooks:
    """Where status replies go: `tf_queue` for a tracked job (no-ops otherwise)."""

    def __init__(self, job: Optional[str]):
        self.job = job

    def status(self, st: dict) -> None:
        if self.job:
            from . import tf_queue

            tf_queue.local_status(self.job, st)

    def moved(self) -> bool:
        if not self.job:
            return False
        from . import tf_queue

        return tf_queue.moved_away(self.job)

    def phase(self, phase: str) -> None:
        if self.job:
            from . import tf_queue

            tf_queue.report(self.job, phase=phase)


async def _wait(c: httpx.AsyncClient, url: str, job_id: str, ctx: Optional[Context], hooks: _Hooks) -> dict:
    """Poll the job (≈ 1 s) until it ends; each new state/stage goes to the log and the MCP client, every
    reply to `hooks` (the queue's progress)."""
    loop = asyncio.get_running_loop()
    last, lost_since = None, None
    while True:
        t0 = loop.time()
        try:
            r = await c.get(f"/v1/jobs/{job_id}", params={"wait": POLL_WAIT_S})
        except httpx.HTTPError as exc:
            lost_since = lost_since or t0
            if loop.time() - lost_since > LOST_AFTER_S:
                raise ToolError(f"lost contact with local TextureFlow at {url} during job {job_id} "
                                f"({type(exc).__name__} for > {LOST_AFTER_S:.0f} s; it may still "
                                f"finish there)") from None
            await asyncio.sleep(2)
            continue
        lost_since = None
        _check(r, url, f"status of job {job_id}")
        st = r.json()
        hooks.status(st)
        now = (st["state"], st.get("stage"), st.get("step"), st.get("queue_position"))
        if now != last:
            last = now
            msg = _describe(st)
            log_progress("texture_flow", f"local job {job_id}: {msg}")
            if ctx:
                try:
                    await ctx.report_progress(progress=st.get("step") or 0, total=st.get("total") or 0,
                                              message=f"TextureFlow (local): {msg}")
                except Exception:  # noqa: BLE001 — best-effort, like the keepalive
                    pass
        if st["state"] in _TERMINAL:
            return st
        if loop.time() - t0 < 0.5:
            await asyncio.sleep(1)  # a server that ignores `wait` must not turn this into a spin


_meta_ok = True  # False once a server refused the `meta` part (older than the queue routes)


async def submit(args: dict, files: dict[str, bytes], meta: Optional[dict] = None) -> dict:
    """`POST /v1/render` → `{job_id, queue_position}`. A 400 is the server's validation error, verbatim."""
    global _meta_ok
    url, token = config()
    parts: list[tuple[str, tuple]] = [("args", (None, json.dumps(args).encode(), "application/json"))]
    parts += [("file", (key, data, "application/octet-stream")) for key, data in files.items()]
    with_meta = bool(meta) and _meta_ok
    async with _client(url, token, httpx.Timeout(30.0, read=60.0)) as c:
        for attempt in (0, 1):
            body = parts + ([("meta", (None, json.dumps(meta).encode(), "application/json"))] if with_meta else [])
            try:
                r = await c.post("/v1/render", files=body)
            except httpx.HTTPError as exc:
                raise ToolError(f"local TextureFlow at {url}: submit failed ({type(exc).__name__})") from None
            if r.status_code == 400 and with_meta and "meta" in _error_text(r) and attempt == 0:
                _meta_ok, with_meta = False, False  # an older server: no queue metadata
                log_progress("texture_flow", "local server predates the `meta` part — sending without it")
                continue
            break
        if r.status_code == 400:
            raise ToolError(_error_text(r))
        _check(r, url, "submit")
        return r.json()


async def collect(job_id: str, ctx: Optional[Context], job: Optional[str] = None) -> dict:
    """Follow server job *job_id* to its end and download it: Modal's `render` dict. Registers the remote
    for a cancel; raises `tf_queue.Moved` when the operator moved the job to Modal meanwhile."""
    url, token = config()
    hooks = _Hooks(job)
    ours = {"cancelled": False}

    async def _cancel() -> Any:
        ours["cancelled"] = True
        return await cancel(job_id)

    jobs.set_remote(f"local|{job_id}", _cancel)
    async with _client(url, token, httpx.Timeout(30.0, read=POLL_WAIT_S + 30.0)) as c:
        st = await with_keepalive(_wait(c, url, job_id, ctx, hooks), ctx, "TextureFlow (local)")
        if st["state"] == "cancelled":
            if hooks.moved():
                from .tf_queue import Moved

                raise Moved(job)
            if ours["cancelled"]:
                raise asyncio.CancelledError  # a cancel is tearing this call down
            raise ToolError(f"local TextureFlow job {job_id} was cancelled on the server")
        if st["state"] == "error":
            raise ToolError(f"local TextureFlow job {job_id} failed: {st.get('error')}")

        hooks.phase("downloading")
        r = await c.get(f"/v1/jobs/{job_id}/result")
        _check(r, url, f"result of job {job_id}")
        res = r.json()
        for name, out in list(res["outputs"].items()):
            f = await c.get(out["url"])
            _check(f, url, f"download of {name}")
            res["outputs"][name] = {"filename": out["filename"], "bytes": f.content,
                                    "mime": out.get("mime") or f.headers.get("content-type")}
    return res


async def render(args: dict, files: dict[str, bytes], ctx: Optional[Context], job: Optional[str] = None,
                 meta: Optional[dict] = None) -> dict:
    """Submit → follow → result → download: Modal's `render(args, files)` dict, rendered locally.

    Without a tracked job, call `health()` first (fail fast); a tracked one waits in `tf_queue.hold_local`.
    """
    sub = await submit(args, files, meta)
    job_id = sub["job_id"]
    log_progress("texture_flow", f"local job {job_id} submitted (queue position {sub.get('queue_position')})")
    if job:
        from . import tf_queue

        tf_queue.local_submitted(job, job_id, sub.get("queue_position"))
    return await collect(job_id, ctx, job)
