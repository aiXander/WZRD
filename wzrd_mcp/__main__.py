"""Entry point for running the WZRD MCP server locally.

Usage:
    python -m wzrd_mcp
    python -m wzrd_mcp --port 8787
    python -m wzrd_mcp --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="WZRD MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")
    args = parser.parse_args()

    from .server import mcp

    mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
