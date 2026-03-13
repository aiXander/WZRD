"""Entry point for running the WZRD MCP server locally.

Usage:
    python -m wzrd_mcp
    python -m wzrd_mcp --port 8787
    python -m wzrd_mcp --host 0.0.0.0 --port 9000

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
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")
    args = parser.parse_args()

    from .server import mcp

    mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
