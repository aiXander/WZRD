"""Terminal logging helpers for WZRD MCP tool calls."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import sys
import time
import traceback
from datetime import datetime
from typing import Annotated, Optional

# Global debug flag — toggled via --debug / --no-debug CLI flag
DEBUG = True

# Parameters that are internal / not client-supplied
_SKIP_PARAMS = {"ctx", "self", "cls"}

# Shared mutable dict for request-level timing.
# Using a single dict (rather than separate ContextVars) so that mutations
# made inside child asyncio tasks propagate back to the middleware — child
# tasks inherit the same dict *reference* even though ContextVar.set() in a
# child wouldn't be visible to the parent.
_request_timing = contextvars.ContextVar("_request_timing", default=None)

# ANSI colors
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _emit(msg: str) -> None:
    """Write one log line to stderr.

    Never stdout: under the stdio transport (Claude Desktop spawns us and speaks
    JSON-RPC over the pipe) stdout *is* the protocol stream, so a stray print
    there breaks framing and the server dies on connect. stderr is the sanctioned
    log channel for stdio servers — Desktop tees it to
    ``~/Library/Logs/Claude/mcp-server-<name>.log``.
    """
    print(msg, file=sys.stderr, flush=True)


def log_call(tool_name: str, args: dict) -> float:
    """Log an incoming tool call with its arguments. Returns start time."""
    if DEBUG:
        _emit(f"\n{_CYAN}{_BOLD}[{_ts()}] ▶ TOOL CALL: {tool_name}{_RESET}")
        for k, v in args.items():
            val = repr(v)
            if len(val) > 120:
                val = val[:117] + "..."
            _emit(f"  {_DIM}{k}={_RESET}{val}")
    return time.time()


def log_progress(tool_name: str, message: str) -> None:
    """Log a progress update during tool execution."""
    if DEBUG:
        _emit(f"{_YELLOW}[{_ts()}]   ⟳ {tool_name}: {message}{_RESET}")


def log_done(tool_name: str, start_time: float, result: object = None) -> None:
    """Log successful tool completion with elapsed time and result."""
    if not DEBUG:
        return
    elapsed = time.time() - start_time
    _emit(f"{_GREEN}[{_ts()}] ✓ {tool_name} completed ({elapsed:.1f}s){_RESET}")
    if result is not None:
        try:
            formatted = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            formatted = repr(result)
        _emit(f"{_DIM}  Result: {formatted}{_RESET}")


def log_error(tool_name: str, error: Exception, start_time: float, *, tb: bool = True) -> None:
    """Log a tool error, with traceback unless it is an expected refusal."""
    if not DEBUG:
        return
    elapsed = time.time() - start_time
    _emit(f"{_RED}[{_ts()}] ✗ {tool_name} failed ({elapsed:.1f}s): {error}{_RESET}")
    if tb:
        _emit(f"{_DIM}{traceback.format_exc()}{_RESET}")


def log_overhead(tool_name: str, req_received: float, tool_start: float,
                 tool_end: float, response_sent: float) -> None:
    """Log the three timing phases for a tool call."""
    if not DEBUG:
        return
    pre_ms = (tool_start - req_received) * 1000
    tool_s = tool_end - tool_start
    post_ms = (response_sent - tool_end) * 1000
    overhead_ms = pre_ms + post_ms
    total_s = response_sent - req_received
    _emit(
        f"{_CYAN}[{_ts()}] ⏱  {tool_name}:  "
        f"pre={pre_ms:.0f}ms  tool={tool_s:.1f}s  post={post_ms:.0f}ms  "
        f"{_BOLD}overhead={overhead_ms:.0f}ms  total={total_s:.1f}s{_RESET}"
    )


def _with_provenance(tool: str, client_args: dict, result: dict, since: float,
                     job: Optional[str]) -> tuple[dict, list[dict], Optional[float]]:
    """Tag + record a call's outputs (see `provenance.py`) and, for a paid
    call, append today's spend. Returns (result, records, recorded cost or
    None). A record failure is logged, never fatal — the output itself is
    already on disk."""
    from . import provenance

    prov = result.pop("_provenance", None)
    records: list[dict] = []
    try:
        records = provenance.record_outputs(tool, client_args, result, since, prov, job)
    except Exception as exc:  # noqa: BLE001
        log_progress(tool, f"render record skipped: {exc}")
    cost = None
    if prov and prov.get("est_usd") is not None:
        cost = float(sum((prov.get("est_usd") or {}).values()))
    if prov and prov.get("paid"):
        result.update(provenance.spend_report())
    return result, records, cost


def _session() -> Optional[str]:
    """The legacy streamable-HTTP `mcp-session-id`, when a client still sends
    one — informational only (a job's `session`). MCP 2026-07-28 clients
    (Claude Code) are sessionless, which is why nothing keys on it."""
    try:
        from fastmcp.server.dependencies import get_http_request

        return get_http_request().headers.get("mcp-session-id")
    except Exception:  # noqa: BLE001 — not an HTTP request
        return None


#: Params `server.tool()` can add to a tool's signature; stripped before the body runs.
_PROJECT_DOC = ("Project (show) this call works on — its files, and what region= resolves against. "
                "Pass it on every call; omitted = the show open in the app.")
_WAIT_DOC = ("false = return at once with a job id (the render runs in the hub); poll jobs(job), "
             "stop it with cancel_job(job). Use for batches and anything over ~1 min. Default true.")
_DEST_DOC = ('Where outputs land: omit or "library" = the Studio target folder, else the media '
             'library\'s inbox/; "project" = the project\'s generated/ (e.g. a segmentation map); '
             'or an absolute folder path.')


def _injected_params(project_arg: bool, dest_arg: bool, render: bool) -> list[inspect.Parameter]:
    from pydantic import Field

    out = []
    if project_arg:
        out.append(inspect.Parameter("project", inspect.Parameter.KEYWORD_ONLY, default=None,
                                     annotation=Annotated[Optional[str], Field(description=_PROJECT_DOC)]))
    if dest_arg:
        out.append(inspect.Parameter("dest", inspect.Parameter.KEYWORD_ONLY, default=None,
                                     annotation=Annotated[Optional[str], Field(description=_DEST_DOC)]))
    if render:
        out.append(inspect.Parameter("wait", inspect.Parameter.KEYWORD_ONLY, default=True,
                                     annotation=Annotated[bool, Field(description=_WAIT_DOC)]))
    return out


def logged_tool(fn, *, project_arg: bool = False, render: bool = False, dest_arg: bool = False):
    """Log a tool call end to end and turn any failure into a `ToolError`.

    Logs the client-supplied params on entry, the result (or the error +
    traceback) on exit, and stamps request timing for `TimingMiddleware`.
    A `ToolError` raised by the body passes through as-is; any other
    exception is logged and re-raised as `ToolError(str(exc))` so the agent
    sees an actionable message instead of FastMCP's masked "internal error".
    Tool bodies therefore need no try/except/log boilerplate of their own.

    Around the body it also sets the call's scope: the `project=` it
    resolves against (`project_arg`), the `dest=` routing of generator
    outputs (`dest_arg`, implied by `render`), and for `render` tools a
    `jobs` row from start to finish plus `wait=false` (the body runs as a
    hub task, see `jobs.detach`). A dict result gets its outputs tagged and
    recorded (`provenance.record_outputs`).

    Applied by `server.tool()` — never stack it by hand.
    """
    from fastmcp.exceptions import ToolError

    from . import jobs, library, project

    dest_arg = dest_arg or render
    extra = _injected_params(project_arg, dest_arg, render)
    extra_names = {p.name for p in extra}

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        scope = {k: kwargs.pop(k) for k in list(kwargs) if k in extra_names}
        client_args = {k: v for k, v in kwargs.items() if k not in _SKIP_PARAMS}
        t0 = log_call(fn.__name__, {**client_args, **{k: v for k, v in scope.items() if v is not None}})
        timing = _request_timing.get(None)
        if timing is not None:
            timing["tool_name"] = fn.__name__
            timing["tool_start"] = time.time()
        tokens = [(project.call_project, project.call_project.set(project.call_project.get()))]
        job = None

        async def body():
            """The call proper: body → records → job end. Runs inline, or as a
            hub task for `wait=false` (it then owns the job's end itself)."""
            try:
                result = await fn(*args, **kwargs)
                records, cost = [], None
                if isinstance(result, dict):
                    result, records, cost = await asyncio.to_thread(
                        _with_provenance, fn.__name__, client_args, result, t0, job)
                if job:
                    jobs.finish(job, ok=True, outputs=[r["path"] for r in records], est_usd=cost)
                    result = {"job": job, **result} if isinstance(result, dict) else result
                log_done(fn.__name__, t0, result if isinstance(result, dict) else None)
                return result
            except asyncio.CancelledError:
                if job:
                    jobs.finish(job, ok=False, error="cancelled", cancelled=True)
                    log_progress(fn.__name__, f"job {job} cancelled")
                raise
            except ToolError as exc:
                if job:
                    jobs.finish(job, ok=False, error=str(exc))
                log_error(fn.__name__, exc, t0, tb=False)
                raise
            except Exception as exc:
                if job:
                    jobs.finish(job, ok=False, error=str(exc))
                log_error(fn.__name__, exc, t0)
                raise ToolError(str(exc)) from exc
            finally:
                if job:
                    jobs.forget(job)

        try:
            if scope.get("project"):
                tokens.append((project.call_project, project.call_project.set(project.slugify(scope["project"]))))
            if dest_arg:
                try:
                    library.check_dest(scope.get("dest"))
                except ValueError as exc:
                    log_error(fn.__name__, exc, t0, tb=False)
                    raise ToolError(str(exc))
                tokens.append((library.current_dest, library.current_dest.set(scope.get("dest"))))
            if render:
                job = jobs.start(fn.__name__, {k: v for k, v in client_args.items() if v is not None},
                                 by=jobs.current_by.get(), session=_session(),
                                 project=project.active_project(), dest=str(library.dest_dir()))
                tokens.append((jobs.current_job, jobs.current_job.set(job)))
                if scope.get("wait") is False:
                    return await jobs.detach(job, body, str(library.dest_dir()))
            return await body()
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
            if timing is not None:
                timing["tool_end"] = time.time()

    if extra:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        at = next((i for i, p in enumerate(params) if p.name == "ctx"), len(params))
        params[at:at] = [p for p in extra if p.name not in sig.parameters]
        # Keyword-only params must follow the positional-or-keyword ones.
        params.sort(key=lambda p: p.kind == inspect.Parameter.KEYWORD_ONLY)
        wrapper.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        wrapper.__annotations__ = {**getattr(fn, "__annotations__", {}),
                                   **{p.name: p.annotation for p in extra}}
    return wrapper


class TimingMiddleware:
    """ASGI middleware that measures MCP server overhead per tool call.

    Sets a context variable when the HTTP request arrives, then after the
    response is fully sent, combines that with tool-start / tool-end
    timestamps (set by ``logged_tool``) to print a timing summary.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        timing = {"request_received": time.time()}
        _request_timing.set(timing)

        async def timed_send(message):
            await send(message)
            # Track the last body chunk as "response sent"
            if message.get("type") == "http.response.body":
                timing["response_sent"] = time.time()

        await self.app(scope, receive, timed_send)

        # Log overhead summary if a tool was executed during this request
        if all(k in timing for k in ("tool_name", "tool_start", "tool_end", "response_sent")):
            log_overhead(
                timing["tool_name"],
                timing["request_received"],
                timing["tool_start"],
                timing["tool_end"],
                timing["response_sent"],
            )
