"""Modal deployment entrypoint for the WZRD MCP server.

Deploy with:
    modal deploy wzrd_mcp/modal_app.py

Deploy without auth:
    WZRD_NO_AUTH=1 modal deploy wzrd_mcp/modal_app.py

Authentication:
    Set WZRD_API_KEY in your Modal secret (eve-secrets-PROD).
    Clients must send: Authorization: Bearer <key>
"""

from __future__ import annotations

import os

import modal

_NO_AUTH_AT_DEPLOY = os.environ.get("WZRD_NO_AUTH", "0") == "1"

app = modal.App("wzrd-mcp")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastmcp>=3.0",
        "httpx>=0.27",
        "boto3>=1.34",
        "scikit-learn>=1.3.0",
        "pillow>=10.0.0",
        "numpy>=2.2.0",
        "opencv-python>=4.8.0,<4.11.0",
        "starlette",
        "fal_client",
    )
    .env({"WZRD_NO_AUTH": "1" if _NO_AUTH_AT_DEPLOY else "0"})
    .add_local_python_source("wzrd", "wzrd_mcp")
)

with image.imports():
    import numpy  # noqa: F401
    import cv2  # noqa: F401
    import sklearn  # noqa: F401
    import PIL  # noqa: F401
    import boto3  # noqa: F401
    import httpx  # noqa: F401
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.types import ASGIApp, Receive, Scope, Send
    from wzrd_mcp.server import mcp


class ApiKeyMiddleware:
    """ASGI middleware that checks for a valid Bearer token."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        api_key = os.environ.get("WZRD_API_KEY", "")
        if not api_key:
            # No key configured — allow all (dev convenience)
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()

        if auth == f"Bearer {api_key}":
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {"error": "Invalid or missing API key. Use: Authorization: Bearer <key>"},
            status_code=401,
        )
        await response(scope, receive, send)


@app.cls(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("eve-secrets-PROD")],
    enable_memory_snapshot=True,
    scaledown_window=120,
)
class McpServer:
    @modal.enter(snap=True)
    def preload(self):
        """Build the ASGI app before snapshotting so restore is instant."""
        self._asgi_app = mcp.http_app(
            transport="streamable-http", stateless_http=True
        )

    @modal.asgi_app()
    def web(self):
        if os.environ.get("WZRD_NO_AUTH", "0") == "1":
            return self._asgi_app
        return ApiKeyMiddleware(self._asgi_app)
