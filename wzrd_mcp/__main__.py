"""Entry point for running the WZRD MCP server locally.

Usage:
    python -m wzrd_mcp                    # the hub: HTTP on 127.0.0.1:8787 (/mcp + /api)
    python -m wzrd_mcp --port 8787
    python -m wzrd_mcp --idle-exit 0      # never exit when idle (default: 3600 s)
    python -m wzrd_mcp --host 0.0.0.0     # expose beyond this machine (no auth!)
    python -m wzrd_mcp --stdio            # stdio (Claude Desktop spawns this)

The HTTP server is the Studio hub (`hub.py`): one per machine, guarded by
`~/WZRD/media/.wzrd/hub.lock` — a second start finds the running hub and
exits 0. The app spawns it; Claude Code connects with
`claude mcp add --transport http wzrd http://127.0.0.1:8787/mcp`.

Set WZRD_API_KEY in .env or environment to require Bearer auth.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from project root if present (no extra deps)."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="WZRD MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")
    parser.add_argument("--debug", default=True, action=argparse.BooleanOptionalAction,
                        help="Enable debug logging (default: True)")
    parser.add_argument("--stdio", action="store_true",
                        help="Serve over stdio instead of HTTP (Claude Desktop "
                             "spawns the server itself; --host/--port ignored)")
    parser.add_argument("--idle-exit", type=float, default=None,
                        help="Exit after this many seconds with no request and no running job "
                             "(default 3600; 0 = never)")
    args = parser.parse_args()

    from . import _log
    _log.DEBUG = args.debug

    from .server import mcp

    if args.stdio:
        # Desktop owns the process and speaks JSON-RPC over our stdout, so
        # nothing may print there — _log writes to stderr, and the FastMCP
        # startup banner is suppressed. TimingMiddleware is ASGI-only and has
        # no meaning here.
        mcp.run(transport="stdio", show_banner=False)
        return

    import atexit
    import sys

    from . import hub
    from ._log import TimingMiddleware, _emit

    running = hub.acquire_lock(args.port)
    if running:
        _emit(f"WZRD hub already running (pid {running.get('pid')}) — nothing to do")
        sys.exit(0)
    atexit.register(hub.release_lock)
    idle = args.idle_exit if args.idle_exit is not None else hub.IDLE_EXIT_S
    hub.start_idle_watch(idle)
    hub.start_trash_flusher()
    from . import library
    library.ensure_projects_root()

    asgi_app = mcp.http_app(transport="streamable-http")
    app = hub.Activity(TimingMiddleware(asgi_app))

    import uvicorn

    class HubServer(uvicorn.Server):
        """Releases hub.lock inside uvicorn's graceful shutdown: uvicorn
        re-raises the SIGTERM it caught once it's done, so code after
        `run()` (and atexit) never runs on a signal."""

        async def shutdown(self, sockets=None):
            hub.STOPPING = True  # open /api/jobs long-polls return now
            try:
                await super().shutdown(sockets)
            finally:
                hub.release_lock()

    try:
        HubServer(uvicorn.Config(app, host=args.host, port=args.port)).run()
    finally:
        hub.release_lock()


if __name__ == "__main__":
    main()
