"""Bearer token authentication middleware for remote (Modal) deployment."""

from __future__ import annotations

import os


class AuthMiddleware:
    """ASGI middleware that validates Authorization: Bearer <token> against WZRD_API_KEY env var.

    If WZRD_API_KEY is not set, all requests are allowed (no auth).
    """

    def __init__(self, app):
        self.app = app
        self.api_key = os.environ.get("WZRD_API_KEY")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.api_key:
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            if not auth_header.startswith("Bearer "):
                await _send_error(send, 401, "Missing Authorization header")
                return

            token = auth_header[len("Bearer "):]
            if token != self.api_key:
                await _send_error(send, 403, "Invalid API key")
                return

        await self.app(scope, receive, send)


async def _send_error(send, status: int, message: str) -> None:
    """Send an HTTP error response."""
    import json

    body = json.dumps({"error": message}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            [b"content-type", b"application/json"],
            [b"content-length", str(len(body)).encode()],
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })
