"""FAL API tool wrappers for the WZRD MCP server: the Kling image→video tools.

(Nano Banana images go to Google directly — see `google_tools.py`.)

Each tool submits work to FAL's queue via fal_client.subscribe (wrapped in
asyncio.to_thread so the MCP event loop is never blocked) and includes
exponential-backoff retry logic for transient errors.

File flow is local-first: results are downloaded straight into the active
project folder (`project.py`) and returned as local paths. Inputs may be local
paths — FAL's workers can't read this disk, so a local input is pushed to
*FAL's own* CDN via `fal_client.upload_file` just in time. That upload is the
one unavoidable outbound copy; it is ephemeral and unrelated to S3.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import fal_client
from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project, provenance
from ._keepalive import with_keepalive
from .file_io import resolve_input_async
from ._log import log_progress
from .server import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_INITIAL_DELAY = 1.0


def _is_retryable(error: Exception) -> bool:
    msg = str(error).lower()
    if any(k in msg for k in ("429", "rate limit", "too many requests")):
        return True
    if any(str(c) in msg for c in range(500, 600)):
        return True
    if any(k in msg for k in ("timeout", "connection", "network", "unavailable")):
        return True
    return False


def _user_error(error: Exception) -> str:
    msg = str(error).lower()
    if "429" in msg or "rate limit" in msg:
        return "FAL rate limit reached. Please try again later."
    if "401" in msg or "unauthorized" in msg:
        return "FAL authentication error. Check FAL_KEY."
    if "403" in msg or "forbidden" in msg:
        return "FAL access denied. Check API permissions."
    if any(str(c) in msg for c in range(500, 600)):
        return "FAL server error. Please try again later."
    if "timeout" in msg:
        return "FAL request timed out. Please try again."
    return str(error)


async def _fal_subscribe(
    endpoint: str, args: dict, tool_name: str = "", ctx: Optional[Context] = None,
) -> dict:
    """Call fal_client.subscribe in a thread with retry + exponential backoff.

    When *ctx* is provided, `with_keepalive` pings it while the job runs so the
    MCP client connection doesn't time out during long FAL jobs.
    """
    if not os.getenv("FAL_KEY"):
        raise ValueError("FAL_KEY environment variable is not set")

    delay = _INITIAL_DELAY
    last_err: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            if tool_name:
                log_progress(tool_name, f"Submitting to FAL ({endpoint})...")

            def _on_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log_entry in update.logs:
                        msg = log_entry["message"]
                        logger.info(msg)
                        if tool_name:
                            log_progress(tool_name, f"FAL: {msg}")

            return await with_keepalive(
                asyncio.to_thread(
                    fal_client.subscribe,
                    endpoint,
                    arguments=args,
                    with_logs=True,
                    on_queue_update=_on_update,
                ),
                ctx, "FAL",
            )

        except Exception as exc:
            last_err = exc
            logger.warning(
                "FAL call to %s failed (attempt %d/%d): %s",
                endpoint,
                attempt + 1,
                _MAX_RETRIES + 1,
                exc,
            )
            if tool_name:
                log_progress(tool_name, f"FAL attempt {attempt + 1} failed: {exc}")
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                if tool_name:
                    log_progress(tool_name, f"Retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise ValueError(_user_error(exc))

    raise ValueError(_user_error(last_err))


def _extract_urls(result: dict) -> list[str]:
    """Pull output URLs from common FAL response shapes."""
    urls: list[str] = []
    if not isinstance(result, dict):
        return urls
    # images: [{url: ...}, ...]
    if "images" in result and isinstance(result["images"], list):
        for item in result["images"]:
            if isinstance(item, dict) and "url" in item:
                urls.append(item["url"])
    # video: {url: ...}
    elif "video" in result and isinstance(result["video"], dict):
        if "url" in result["video"]:
            urls.append(result["video"]["url"])
    # direct url
    elif "url" in result:
        urls.append(result["url"])
    return urls


# ---------------------------------------------------------------------------
# Local file -> FAL-visible URL
# ---------------------------------------------------------------------------
async def _to_fal_url(value: Optional[str]) -> Optional[str]:
    """Make *value* fetchable by FAL's workers.

    Passes http(s) URLs through untouched. Anything else — a local path, a
    base64/data blob — is materialised locally and uploaded to FAL's CDN.
    """
    if not value or value.startswith(("http://", "https://")):
        return value

    try:
        local = await resolve_input_async(value)
    except FileNotFoundError:
        raise ToolError(f"input is neither a URL nor a file on this machine: {value}")
    return await asyncio.to_thread(fal_client.upload_file, local)


async def _to_fal_urls(values: list[Optional[str]]) -> list[Optional[str]]:
    """`_to_fal_url` across a list, uploading any local inputs concurrently."""
    return list(await asyncio.gather(*(_to_fal_url(v) for v in values)))


# FAL list prices (fal.ai model pages, 2026-10-10).
_KLING_V25_5S = 0.35
_KLING_V25_PER_EXTRA_S = 0.07
_KLING_V3_PER_S = 0.084
_KLING_V3_PER_S_AUDIO = 0.126


def _with_cost(result: dict, usd: float, endpoint: str, region_info: dict) -> dict:
    """Attach the provenance (cost on the video, model, region) `logged_tool` writes."""
    video = os.path.realpath(result["output_video"])
    result["_provenance"] = {"paid": True, "est_usd": {video: usd},
                             "args": {"model": endpoint}, **region_info}
    return result


def _nearest_aspect(image: str, choices: tuple[str, ...]) -> str:
    """The choice nearest a local image's aspect (log-ratio); first choice for URLs.

    A video model forced to a far-off aspect crops or squashes its start frame;
    the nearest one keeps a tall region's keyframe tall.
    """
    import math

    from wzrd.utils import load_upright

    path = os.path.expanduser(image)
    if not os.path.isfile(path):
        return choices[0]
    w, h = load_upright(path).size  # as shot: an iPhone portrait is tall

    def ratio(c: str) -> float:
        a, b = c.split(":")
        return float(a) / float(b)

    return min(choices, key=lambda c: abs(math.log((w / h) / ratio(c))))


# ---------------------------------------------------------------------------
# Common FAL tool pipeline
# ---------------------------------------------------------------------------
async def _run_fal_tool(
    tool_name: str,
    endpoint: str,
    fal_args: dict,
    output_type: str,
    response_info: dict,
    ctx: Optional[Context] = None,
) -> dict:
    """Subscribe to a FAL endpoint and download the result into the project.

    Args:
        tool_name: Name for logging.
        endpoint: FAL endpoint path (e.g. "fal-ai/kling-video/v3/...").
        fal_args: Arguments dict sent to FAL.
        output_type: "video" (single .mp4) or "images" (one or more .jpg).
        response_info: Extra metadata included in the response "info" dict.
        ctx: Optional MCP context for progress reporting.
    """
    result = await _fal_subscribe(endpoint, fal_args, tool_name=tool_name, ctx=ctx)

    urls = _extract_urls(result)
    if not urls:
        raise ToolError(f"No {output_type} URL(s) in FAL response")

    if output_type == "video":
        log_progress(tool_name, "Downloading video from FAL...")
        path = await project.download_async(urls[0], tool=tool_name, kind=project.GENERATED, ext=".mp4")
        return {"output_video": path, "info": response_info}

    log_progress(tool_name, f"Downloading {len(urls)} image(s) from FAL...")
    paths = list(await asyncio.gather(*[
        project.download_async(u, tool=tool_name, kind=project.GENERATED, ext=".jpg") for u in urls
    ]))
    return {"output_images": paths, "info": {**response_info, "num_images": len(paths)}}


# ---------------------------------------------------------------------------
# Tool: Kling v2.5 Turbo Pro Image-to-Video
# ---------------------------------------------------------------------------
@tool(title="Image to video (Kling v2.5)", tags={"generate", "video"}, open_world=True)
async def kling_v25_image_to_video(
    prompt: str,
    image_url: str,
    duration: str = "5",
    tail_image_url: Optional[str] = None,
    negative_prompt: str = "blur, distort, and low quality",
    cfg_scale: float = 0.5,
    region: Optional[str] = None,
    allow_over_cap: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Animate a start frame (optionally into an end frame) as a 5 or 10 s video with Kling v2.5 Turbo Pro via FAL.

    For projection, write the prompt per skill `video-prompting` (static
    camera, unchanged dark background, motion described plainly).

    Args:
        prompt: Text description of the desired video motion and content.
        image_url: URL *or local file path* of the image to use as the first frame.
        duration: Video length in seconds. Choices: "5" or "10".
        tail_image_url: Optional URL or local path of an image for the last frame (start→end transition).
        negative_prompt: Things to avoid in the generation.
        cfg_scale: Guidance scale 0.0-1.0. Lower = more creative, higher = closer to prompt.
        region: Optional pack layer id this output is made for (e.g. "pillar_left"). Recorded in its provenance sidecar with that layer's box, so the clip grid files it in that region's row.
        allow_over_cap: Only after Xander OKs it: run even though it takes the project past its spend cap.
    """
    region_info = provenance.region_meta(region) if region else {}
    usd = _KLING_V25_5S + _KLING_V25_PER_EXTRA_S * max(0, int(duration) - 5)
    provenance.check_spend(usd, allow_over_cap)
    image_url, tail_image_url = await _to_fal_urls([image_url, tail_image_url])

    fal_args = {
        "prompt": prompt,
        "image_url": image_url,
        "duration": duration,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
    }
    if tail_image_url:
        fal_args["tail_image_url"] = tail_image_url

    endpoint = "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"
    result = await _run_fal_tool(
        "kling_v25_image_to_video", endpoint, fal_args, output_type="video",
        response_info={"duration": duration}, ctx=ctx,
    )
    return _with_cost(result, usd, endpoint, region_info)


# ---------------------------------------------------------------------------
# Tool: Kling v3 Image-to-Video
# ---------------------------------------------------------------------------
@tool(title="Image to video (Kling v3)", tags={"generate", "video"}, open_world=True)
async def kling_v3_image_to_video(
    prompt: str,
    start_image_url: str,
    duration: str = "5",
    generate_audio: bool = False,
    end_image_url: Optional[str] = None,
    negative_prompt: str = "blur, distort, and low quality",
    cfg_scale: float = 0.5,
    aspect_ratio: Optional[str] = None,
    region: Optional[str] = None,
    allow_over_cap: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Generate a 3-15 second video from a starting image using Kling v3.0 Standard via FAL.

    Creates high-quality cinematic videos with optional native audio from a single
    starting image. Optionally provide an end image for start→end transitions.

    Args:
        prompt: Text description of the desired video motion and content.
        start_image_url: URL *or local file path* of the image to use as the first frame.
        duration: Video length in seconds. Choices: "3" through "15".
        generate_audio: Generate native audio for the video (Chinese/English). Increases cost ~50%.
        end_image_url: Optional URL or local path of an image for the last frame (start→end transition).
        negative_prompt: Things to avoid in the generation.
        cfg_scale: Guidance scale 0.0-1.0. Lower = more creative, higher = closer to prompt.
        aspect_ratio: "16:9", "9:16" or "1:1". Default: the one nearest the start image's aspect.
        region: Optional pack layer id this output is made for (e.g. "pillar_left"). Recorded in its provenance sidecar with that layer's box, so the clip grid files it in that region's row.
        allow_over_cap: Only after Xander OKs it: run even though it takes the project past its spend cap.
    """
    region_info = provenance.region_meta(region) if region else {}
    usd = int(duration) * (_KLING_V3_PER_S_AUDIO if generate_audio else _KLING_V3_PER_S)
    provenance.check_spend(usd, allow_over_cap)
    aspect_ratio = aspect_ratio or _nearest_aspect(start_image_url, ("16:9", "9:16", "1:1"))
    start_image_url, end_image_url = await _to_fal_urls([start_image_url, end_image_url])

    fal_args = {
        "prompt": prompt,
        "start_image_url": start_image_url,
        "duration": duration,
        "generate_audio": generate_audio,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "aspect_ratio": aspect_ratio,
    }
    if end_image_url:
        fal_args["end_image_url"] = end_image_url

    endpoint = "fal-ai/kling-video/v3/standard/image-to-video"
    result = await _run_fal_tool(
        "kling_v3_image_to_video", endpoint, fal_args, output_type="video",
        response_info={"duration": duration, "aspect_ratio": aspect_ratio}, ctx=ctx,
    )
    return _with_cost(result, usd, endpoint, region_info)
