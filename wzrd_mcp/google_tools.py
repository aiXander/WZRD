"""Google API tool wrappers for the WZRD MCP server: Nano Banana 2.1 images.

Calls Google directly with the `google-genai` SDK — reference images go inline
in the request as bytes (no upload step, no queue polling) and the result lands
in the active project folder as a local path.

Backend: the Gemini API (`GEMINI_API_KEY`) first; Vertex AI (`VERTEX_API_KEY`)
only when the Gemini key is unset. Keys are always passed explicitly — the SDK's
own env auto-detection would let a stray `GOOGLE_API_KEY` silently win.
"""

from __future__ import annotations

import asyncio
import functools
import io
import os
from pathlib import Path
from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from google import genai
from google.genai import errors, types
from PIL import Image

from . import jobs, library, project, provenance
from ._keepalive import with_keepalive
from ._log import log_progress
from .file_io import resolve_input_async
from .server import tool

# Runs Nano Banana 2.1. It replaced Nano Banana Pro on 2026-10-09 after a blind
# shootout on WZRD tasks (scripts/fal_shootout.py): better on story continuity,
# keyframe pairs and keeping a projection surface untouched; ~4× cheaper
# (token-billed, ~$0.04 at 1K) and ~2× faster. It moved from FAL to Google
# direct the same day (scripts/google_nb_probe.py: equal quality, no upload,
# no FAL markup). The tool name is model-agnostic so the next swap is _MODEL.
_MODEL = "gemini-nano-banana-2.1"
_VERTEX_PROJECT_DEFAULT = "gen-lang-client-0837456283"

_MAX_REFS = 14
_REF_MAX_SIDE = 2048  # larger refs are shrunk in memory: payload size costs latency
_FILTER_RETRIES = 2

# Finish/block reasons that mean "a content filter fired". Often flaky, so retried.
_FILTER_REASONS = {
    "SAFETY", "IMAGE_SAFETY", "PROHIBITED_CONTENT", "IMAGE_PROHIBITED_CONTENT",
    "BLOCKLIST", "SPII",
}

# USD per 1M tokens (Google list price, standard tier, 2026-10).
_USD_PER_M_INPUT = 1.50
_USD_PER_M_TEXT_OUT = 7.50  # text + thought tokens
_USD_PER_M_IMAGE_OUT = 30.0
# Pre-call estimate for the spend cap (the real cost comes from usage after).
_USD_PER_IMAGE = {"1K": 0.034, "2K": 0.050, "4K": 0.113}
_USD_PER_REF = 0.002


class _NoImage(Exception):
    """The response carried no final image; *reason* is the filter/finish reason."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@functools.cache
def _client() -> tuple[genai.Client, str]:
    """(client, backend name). Gemini API key first, Vertex as the fallback."""
    # The SDK retries 408/429/5xx and httpx transport errors with backoff.
    http = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=4))
    if key := os.getenv("GEMINI_API_KEY"):
        return genai.Client(api_key=key, http_options=http), "gemini"
    if key := os.getenv("VERTEX_API_KEY"):
        # Both project and location="global" are required: no location → 404
        # (us-central1), no project → 403.
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", _VERTEX_PROJECT_DEFAULT)
        return genai.Client(
            vertexai=True, api_key=key, project=project_id, location="global",
            http_options=http,
        ), "vertex"
    raise ToolError("No Google key: set GEMINI_API_KEY (preferred) or VERTEX_API_KEY in .env")


def _load_ref(path: str) -> types.Blob:
    """Read one reference image; shrink + re-encode to JPEG if it's oversized
    or in a format Google doesn't take. Never writes the re-encoded copy."""
    with open(path, "rb") as f:
        data = f.read()
    img = Image.open(io.BytesIO(data))
    fmt = (img.format or "").upper()
    if max(img.size) <= _REF_MAX_SIDE and fmt in ("JPEG", "PNG", "WEBP"):
        return types.Blob(mime_type=f"image/{fmt.lower()}", data=data)
    img.thumbnail((_REF_MAX_SIDE, _REF_MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return types.Blob(mime_type="image/jpeg", data=buf.getvalue())


async def _contents(prompt: str, refs: list[str]) -> list[types.Part]:
    """`image_1:` label + image part per ref (1-based — the model's own text
    numbers inputs from 0, so the explicit labels matter), prompt last."""
    try:
        paths = await asyncio.gather(*(resolve_input_async(r) for r in refs))
    except FileNotFoundError as exc:
        raise ToolError(f"reference image is neither a URL nor a file on this machine: {exc}")
    blobs = await asyncio.gather(*(asyncio.to_thread(_load_ref, p) for p in paths))
    parts: list[types.Part] = []
    for i, blob in enumerate(blobs, 1):
        parts += [types.Part(text=f"image_{i}:"), types.Part(inline_data=blob)]
    return parts + [types.Part(text=prompt)]


def _final_image(resp: types.GenerateContentResponse) -> types.Blob:
    """The last non-thought image part. Vertex at `medium` thinking emits a
    draft image (flagged `thought`) before the final — drafts are skipped."""
    cand = (resp.candidates or [None])[0]
    parts = (cand.content.parts if cand and cand.content else None) or []
    images = [p.inline_data for p in parts if p.inline_data and not p.thought]
    if images:
        return images[-1]

    finish = getattr(cand, "finish_reason", None)
    block = getattr(resp.prompt_feedback, "block_reason", None)
    reason = (getattr(block, "name", None) or getattr(finish, "name", None) or "NO_IMAGE")
    text = " ".join(p.text for p in parts if p.text and not p.thought).strip()
    detail = "; ".join(filter(None, [
        f"finish_reason={finish.name}" if finish else "",
        f"block_reason={block.name}" if block else "",
        f"model said: {text[:300]}" if text else "",
    ]))
    raise _NoImage(reason, detail)


def _est_usd(resp: types.GenerateContentResponse) -> float:
    """Cost of one call from its usage metadata, at Google list price."""
    u = resp.usage_metadata
    if not u:
        return 0.0
    image_out = sum(
        d.token_count or 0 for d in (u.candidates_tokens_details or [])
        if d.modality == types.MediaModality.IMAGE
    )
    text_out = (u.candidates_token_count or 0) - image_out + (u.thoughts_token_count or 0)
    return (
        (u.prompt_token_count or 0) * _USD_PER_M_INPUT
        + text_out * _USD_PER_M_TEXT_OUT
        + image_out * _USD_PER_M_IMAGE_OUT
    ) / 1e6


async def _generate_one(
    client: genai.Client, contents: list[types.Part], config: types.GenerateContentConfig,
) -> tuple[types.Blob, float]:
    """One image. Content-filter misses are retried immediately (they're often
    flaky); transport retries happen inside the SDK."""
    spent = 0.0
    for attempt in range(_FILTER_RETRIES + 1):
        try:
            resp = await client.aio.models.generate_content(
                model=_MODEL, contents=contents, config=config,
            )
        except errors.APIError as exc:
            if exc.code in (401, 403):
                raise ToolError(
                    f"Google rejected the key ({exc.code}): check GEMINI_API_KEY / "
                    f"VERTEX_API_KEY. {exc.message}"
                )
            raise ToolError(f"Google API error {exc.code}: {exc.message}")
        spent += _est_usd(resp)
        try:
            return _final_image(resp), spent
        except _NoImage as exc:
            if exc.reason in _FILTER_REASONS and attempt < _FILTER_RETRIES:
                log_progress("nano_banana", f"content filter ({exc.reason}), retrying…")
                continue
            raise ToolError(f"Nano Banana returned no image — {exc}")
    raise AssertionError("unreachable")


_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def _estimate(args: dict) -> float:
    """Pre-call cost: tier price × images (+ refs). Without a resolution, the
    1K price — a canvas project may pick 2K, which the call itself prices."""
    tier = str(args.get("resolution") or "1K").upper()
    refs = len(args.get("image_urls") or [])
    return int(args.get("num_images") or 1) * (_USD_PER_IMAGE.get(tier, 0.113) + _USD_PER_REF * refs)


jobs.ESTIMATORS["nano_banana"] = _estimate


@tool(title="Nano Banana image", tags={"generate", "image"}, open_world=True, project_arg=True, render=True)
async def nano_banana(
    prompt: str,
    image_urls: Optional[list[str]] = None,
    num_images: int = 1,
    resolution: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    enable_web_search: bool = False,
    seed: int = -1,
    thinking_level: str = "medium",
    region: Optional[str] = None,
    allow_over_cap: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Generate or edit an image with Nano Banana 2.1 (Google). Keyframes on a surface, surface masking, segmentation maps, character poses.

    No image_urls = text-to-image. With image_urls, each reference is labelled
    image_1, image_2, … in list order and the prompt addresses them by those
    names, e.g. "Place the character from image_2 on the wall in image_1".
    Before writing a prompt, load skill `image-prompting` (background lock,
    reference handling, additive-light rules, anti-slop formula, templates).

    Args:
        prompt: Text description of the desired image(s). When reference images
            are provided, use image_1, image_2, … image_N in the prompt to
            refer to specific input images by their position in the list.
        image_urls: Optional list of reference images (up to 14, best ≤10 —
            quality drops with more), each a URL *or a local file path*. Each
            entry becomes a named reference: the first is image_1, the second
            is image_2, and so on. The prompt should explicitly mention these
            references to control how each image influences the output.
        num_images: Images per call (1-4). For variations, prefer parallel calls with num_images=1 and slightly different prompts.
        resolution: "1K", "2K" or "4K" (billed per output token: ~$0.034 / $0.050 / $0.113 per image). Default: the project canvas's tier, else "1K".
        aspect_ratio: "auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "4:1", "1:4", "8:1", "1:8". Default: "auto" with references (image_1's aspect: a canvas surface stays the canvas, a region_surface_image stays its region's box), else the project canvas's aspect, else "1:1".
        enable_web_search: Use Google Search grounding for current events or real-world references.
        seed: Random seed (0-2147483647). It nudges the composition but does not reproduce an image exactly.
        thinking_level: How much the model reasons before drawing. "minimal" (~30% faster, fine for simple edits), "medium" (default), or "high".
        region: Optional pack layer id this output is made for (e.g. "pillar_left"). Recorded in the render's record with that layer's box (resolved against this call's project).
        allow_over_cap: Only after Xander OKs it: run even though it takes the project past its spend cap.
    """
    image_urls = image_urls or []
    canvas = project.canvas()
    canvas_aspect, canvas_tier = project.canvas_aspect(canvas) if canvas else ("1:1", "1K")
    aspect_ratio = aspect_ratio or ("auto" if image_urls else canvas_aspect)
    resolution = resolution or canvas_tier
    if len(image_urls) > _MAX_REFS:
        raise ToolError(f"at most {_MAX_REFS} reference images (got {len(image_urls)}); best ≤10")
    if not 1 <= num_images <= 4:
        raise ToolError(f"num_images must be 1-4 (got {num_images})")
    resolution = resolution.upper()
    region_info = provenance.region_meta(region) if region else {}
    provenance.check_spend(
        _estimate({"num_images": num_images, "resolution": resolution, "image_urls": image_urls}),
        allow_over_cap,
    )
    client, backend = _client()

    image_config = types.ImageConfig(image_size=resolution)
    if aspect_ratio and aspect_ratio != "auto":
        image_config.aspect_ratio = aspect_ratio
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=image_config,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        seed=seed if seed >= 0 else None,
        tools=[types.Tool(google_search=types.GoogleSearch())] if enable_web_search else None,
    )
    contents = await _contents(prompt, image_urls)

    log_progress("nano_banana", f"Calling Google ({backend}, {_MODEL}) × {num_images}…")
    results = await with_keepalive(
        asyncio.gather(
            *(_generate_one(client, contents, config) for _ in range(num_images)),
            return_exceptions=True,
        ),
        ctx, "Google",
    )
    ok = [r for r in results if not isinstance(r, BaseException)]
    failed = [r for r in results if isinstance(r, BaseException)]
    if not ok:
        raise failed[0]

    paths, costs = [], {}
    for blob, usd in ok:
        path = library.output_path("nano_banana", _EXT.get(blob.mime_type, ".png"))
        with open(path, "wb") as f:
            f.write(blob.data)
        paths.append(path)
        costs[str(Path(path).resolve())] = usd

    info = {
        "resolution": resolution, "aspect_ratio": aspect_ratio,
        "mode": "edit" if image_urls else "txt2img",
        "num_images": len(paths), "backend": backend,
        "thinking_level": thinking_level,
        "est_usd": round(sum(usd for _, usd in ok), 4),
    }
    if failed:
        info["failed"] = [str(e) for e in failed]
    return {
        "output_images": paths, "info": info,
        "_provenance": {"paid": True, "est_usd": costs,
                        "args": {"model": _MODEL, "backend": backend}, **region_info},
    }
