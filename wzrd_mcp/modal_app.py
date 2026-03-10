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


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("eve-secrets-PROD")],
)
@modal.asgi_app()
def web():
    from wzrd_mcp.server import mcp

    return mcp.http_app(transport="streamable-http", stateless_http=True)
