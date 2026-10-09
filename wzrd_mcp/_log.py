"""Terminal logging helpers for WZRD MCP tool calls."""

from __future__ import annotations

import contextvars
import functools
import json
import sys
import time
import traceback
from datetime import datetime

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


def logged_tool(fn):
    """Log a tool call end to end and turn any failure into a `ToolError`.

    Logs the client-supplied params on entry, the result (or the error +
    traceback) on exit, and stamps request timing for `TimingMiddleware`.
    A `ToolError` raised by the body passes through as-is; any other
    exception is logged and re-raised as `ToolError(str(exc))` so the agent
    sees an actionable message instead of FastMCP's masked "internal error".
    Tool bodies therefore need no try/except/log boilerplate of their own.

    Applied by `server.tool()` — never stack it by hand.
    """
    from fastmcp.exceptions import ToolError

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        client_args = {k: v for k, v in kwargs.items() if k not in _SKIP_PARAMS}
        t0 = log_call(fn.__name__, client_args)
        timing = _request_timing.get(None)
        if timing is not None:
            timing["tool_name"] = fn.__name__
            timing["tool_start"] = time.time()
        try:
            result = await fn(*args, **kwargs)
            log_done(fn.__name__, t0, result if isinstance(result, dict) else None)
            return result
        except ToolError as exc:
            log_error(fn.__name__, exc, t0, tb=False)
            raise
        except Exception as exc:
            log_error(fn.__name__, exc, t0)
            raise ToolError(str(exc)) from exc
        finally:
            if timing is not None:
                timing["tool_end"] = time.time()

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
