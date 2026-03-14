"""Terminal logging helpers for WZRD MCP tool calls."""

from __future__ import annotations

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
        bound = sig.bind(*args, **kwargs)
        client_args = {
            k: v for k, v in bound.arguments.items() if k not in _SKIP_PARAMS
        }
        log_call(fn.__name__, client_args)
        return await fn(*args, **kwargs)

    return wrapper
