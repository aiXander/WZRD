"""Terminal logging helpers for WZRD MCP tool calls."""

from __future__ import annotations

import time
import traceback
from datetime import datetime

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
    print(f"\n{_CYAN}{_BOLD}[{_ts()}] ▶ TOOL CALL: {tool_name}{_RESET}")
    for k, v in args.items():
        val = repr(v)
        if len(val) > 120:
            val = val[:117] + "..."
        print(f"  {_DIM}{k}={_RESET}{val}")
    return time.time()


def log_progress(tool_name: str, message: str) -> None:
    """Log a progress update during tool execution."""
    print(f"{_YELLOW}[{_ts()}]   ⟳ {tool_name}: {message}{_RESET}")


def log_done(tool_name: str, start_time: float) -> None:
    """Log successful tool completion with elapsed time."""
    elapsed = time.time() - start_time
    print(f"{_GREEN}[{_ts()}] ✓ {tool_name} completed ({elapsed:.1f}s){_RESET}")


def log_error(tool_name: str, error: Exception, start_time: float) -> None:
    """Log a tool error with traceback."""
    elapsed = time.time() - start_time
    print(f"{_RED}[{_ts()}] ✗ {tool_name} failed ({elapsed:.1f}s): {error}{_RESET}")
    print(f"{_DIM}{traceback.format_exc()}{_RESET}")
