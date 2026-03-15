"""Terminal logging helpers for WZRD MCP tool calls."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
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


def log_call(tool_name: str, args: dict) -> float:
    """Log an incoming tool call with its arguments. Returns start time."""
    if DEBUG:
        print(f"\n{_CYAN}{_BOLD}[{_ts()}] ▶ TOOL CALL: {tool_name}{_RESET}")
        for k, v in args.items():
            val = repr(v)
            if len(val) > 120:
                val = val[:117] + "..."
            print(f"  {_DIM}{k}={_RESET}{val}")
    return time.time()


def log_progress(tool_name: str, message: str) -> None:
    """Log a progress update during tool execution."""
    if DEBUG:
        print(f"{_YELLOW}[{_ts()}]   ⟳ {tool_name}: {message}{_RESET}")


def log_done(tool_name: str, start_time: float, result: object = None) -> None:
    """Log successful tool completion with elapsed time and result."""
    if not DEBUG:
        return
    elapsed = time.time() - start_time
    print(f"{_GREEN}[{_ts()}] ✓ {tool_name} completed ({elapsed:.1f}s){_RESET}")
    if result is not None:
        try:
            formatted = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            formatted = repr(result)
        print(f"{_DIM}  Result: {formatted}{_RESET}")


def log_error(tool_name: str, error: Exception, start_time: float) -> None:
    """Log a tool error with traceback."""
    if not DEBUG:
        return
    elapsed = time.time() - start_time
    print(f"{_RED}[{_ts()}] ✗ {tool_name} failed ({elapsed:.1f}s): {error}{_RESET}")
    print(f"{_DIM}{traceback.format_exc()}{_RESET}")


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
    print(
        f"{_CYAN}[{_ts()}] ⏱  {tool_name}:  "
        f"pre={pre_ms:.0f}ms  tool={tool_s:.1f}s  post={post_ms:.0f}ms  "
        f"{_BOLD}overhead={overhead_ms:.0f}ms  total={total_s:.1f}s{_RESET}"
    )


def logged_tool(fn):
    """Decorator that auto-logs all client-supplied params on entry.

    Stacks with @mcp.tool() — apply this *below* @mcp.tool() so that
    FastMCP sees the original signature for schema generation.

    Usage:
        @mcp.tool()
        @logged_tool
        async def my_tool(arg1: str, arg2: int = 5, ctx=None) -> dict:
            ...
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        # Only log params actually supplied by the caller, not defaults
        client_args = {
            k: v for k, v in kwargs.items() if k not in _SKIP_PARAMS
        }
        log_call(fn.__name__, client_args)
        timing = _request_timing.get(None)
        if timing is not None:
            timing["tool_name"] = fn.__name__
            timing["tool_start"] = time.time()
        try:
            return await fn(*args, **kwargs)
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
