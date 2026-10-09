"""MCP keepalive for long-running remote calls.

MCP clients drop a tool call that stays silent too long, so while a remote job
runs we send a progress notification every few seconds. Shared by the FAL
tools (Kling) and the Google tools (Nano Banana).
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Optional, TypeVar

from fastmcp import Context

T = TypeVar("T")

_EVERY_S = 15.0


async def with_keepalive(awaitable: Awaitable[T], ctx: Optional[Context], what: str) -> T:
    """Await *awaitable*, pinging *ctx* with progress every 15 s until it finishes.

    Returns as soon as the work completes (no rounding up to the next tick).
    Cancelling the caller cancels the work.
    """
    task = asyncio.ensure_future(awaitable)
    tick = 0
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=_EVERY_S)
            if done:
                return task.result()
            tick += 1
            if ctx:
                try:
                    await ctx.report_progress(
                        progress=tick, total=0,
                        message=f"Waiting for {what} ({tick * _EVERY_S:.0f}s elapsed)…",
                    )
                except Exception:
                    pass  # best-effort keepalive
    except asyncio.CancelledError:
        task.cancel()
        raise
