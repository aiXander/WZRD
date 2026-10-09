"""Text-only tool results — stop shipping every payload twice.

FastMCP emits a dict-returning tool's result as BOTH a JSON text block and a
parsed `structuredContent` duplicate: the whole payload twice in the model's
context on every call. Wrapping the body with `jsonify_result` makes the
runtime value one compact JSON string; paired with
`@mcp.tool(output_schema=None)` (no schema → nothing to wrap the string into),
FastMCP ships exactly ONE text block. Same pattern as the motherbrain MCP.

Non-dict results pass through untouched: engine tools return a caption string
or a `[caption, Image]` list, which FastMCP already renders as plain content.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def jsonify_result(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = await fn(*args, **kwargs)
        return compact_json(result) if isinstance(result, dict) else result

    # Same parameters as the body (FastMCP's argument schema), but an `Any`
    # return: the runtime value is now a str, and a `dict` annotation would make
    # pydantic warn on every call while serializing it.
    wrapper.__signature__ = inspect.signature(fn).replace(return_annotation=Any)  # type: ignore[attr-defined]
    wrapper.__annotations__ = {**getattr(fn, "__annotations__", {}), "return": Any}
    return wrapper
