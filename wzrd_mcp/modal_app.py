"""Modal deployment entrypoint for the WZRD MCP server.

Deploy with:
    modal deploy wzrd_mcp/modal_app.py
"""

from __future__ import annotations

import modal

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
    )
    .add_local_python_source("wzrd", "wzrd_mcp")
)

with image.imports():
    import numpy  # noqa: F401
    import cv2  # noqa: F401
    import sklearn  # noqa: F401
    import PIL  # noqa: F401
    import boto3  # noqa: F401
    import httpx  # noqa: F401
    from wzrd_mcp.server import mcp


@app.cls(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("eve-secrets-PROD")],
    enable_memory_snapshot=True,
    container_idle_timeout=120,
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
        return self._asgi_app
