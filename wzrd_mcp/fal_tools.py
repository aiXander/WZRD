"""FAL API tool wrappers for the WZRD MCP server.

Each tool submits work to FAL's queue via fal_client.subscribe (wrapped in
asyncio.to_thread so the MCP event loop is never blocked) and includes
exponential-backoff retry logic for transient errors.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Optional

import fal_client
import httpx
from fastmcp import Context
from fastmcp.exceptions import ToolError

from .file_io import upload_async
from ._log import log_call, log_progress, log_done, log_error, logged_tool
from .server import mcp, get_timeout

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

    When *ctx* is provided, sends periodic MCP progress notifications as
    keepalives so the client connection doesn't time out during long FAL jobs.
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

            # Run subscribe in a thread; meanwhile send keepalive progress
            # notifications every 15s so the MCP client doesn't drop us.
            subscribe_task = asyncio.ensure_future(
                asyncio.to_thread(
                    fal_client.subscribe,
                    endpoint,
                    arguments=args,
                    with_logs=True,
                    on_queue_update=_on_update,
                )
            )

            tick = 0
            while not subscribe_task.done():
                await asyncio.sleep(15)
                if subscribe_task.done():
                    break
                tick += 1
                if ctx:
                    try:
                        await ctx.report_progress(
                            progress=tick,
                            total=0,
                            message=f"Waiting for FAL ({tick * 15}s elapsed)…",
                        )
                    except Exception:
                        pass  # best-effort keepalive

            result = await subscribe_task
            return result

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


async def _download_to_tmp(url: str, suffix: str = "") -> str:
    """Download a URL to a temp file (async)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tool: Kling v3 Image-to-Video
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("kling_v3_image_to_video"))
@logged_tool
async def kling_v3_image_to_video(
    prompt: str,
    start_image_url: str,
    duration: str = "5",
    generate_audio: bool = False,
    end_image_url: str = "",
    negative_prompt: str = "blur, distort, and low quality",
    cfg_scale: float = 0.5,
    aspect_ratio: str = "16:9",
    ctx: Optional[Context] = None,
) -> dict:
    """Generate a 3-15 second video from a starting image using Kling v3.0 Standard via FAL.

    Creates high-quality cinematic videos with optional native audio from a single
    starting image. Optionally provide an end image for start→end transitions.

    Args:
        prompt: Text description of the desired video motion and content.
        start_image_url: URL of the image to use as the first frame.
        duration: Video length in seconds. Choices: "3" through "15".
        generate_audio: Generate native audio for the video (Chinese/English). Increases cost ~50%.
        end_image_url: Optional URL of an image for the last frame (start→end transition).
        negative_prompt: Things to avoid in the generation.
        cfg_scale: Guidance scale 0.0-1.0. Lower = more creative, higher = closer to prompt.
        aspect_ratio: Output aspect ratio. Choices: "16:9", "9:16", "1:1".
    """
    _name = "kling_v3_image_to_video"
    t0 = time.time()
    try:
        fal_args: dict = {
            "prompt": prompt,
            "start_image_url": start_image_url,
            "duration": duration,
            "generate_audio": generate_audio,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
            "aspect_ratio": aspect_ratio,
        }
        if end_image_url != "":
            fal_args["end_image_url"] = end_image_url

        result = await _fal_subscribe(
            "fal-ai/kling-video/v3/standard/image-to-video", fal_args,
            tool_name=_name, ctx=ctx,
        )

        urls = _extract_urls(result)
        if not urls:
            err = "No video URL in FAL response"
            log_error(_name, ValueError(err), t0)
            raise ToolError(err)

        log_progress(_name, "Downloading video from FAL...")
        if ctx:
            try:
                await ctx.report_progress(progress=90, total=100, message="Downloading video from FAL…")
            except Exception:
                pass
        video_tmp = await _download_to_tmp(urls[0], suffix=".mp4")

        log_progress(_name, "Uploading to S3...")
        if ctx:
            try:
                await ctx.report_progress(progress=95, total=100, message="Uploading to S3…")
            except Exception:
                pass
        uploaded_url = await upload_async(video_tmp)

        response = {
            "output_video": uploaded_url,
            "info": {
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            },
        }
        log_done(_name, t0, response)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool: Nano Banana Pro (txt2img + img2img unified)
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("nano_banana_pro"))
@logged_tool
async def nano_banana_pro(
    prompt: str,
    image_urls: list[str] = [],
    num_images: int = 1,
    resolution: str = "1K",
    aspect_ratio: str = "1:1",
    enable_web_search: bool = False,
    seed: int = -1,
    ctx: Optional[Context] = None,
) -> dict:
    """Generate or edit images using Google's Nano Banana Pro model via FAL.

    When no image_urls are provided this runs text-to-image generation.
    When image_urls are provided it switches to editing/reference mode: the
    images become named references that the prompt can address as image_1,
    image_2, … image_N (matching the order they appear in the list, up to 14).

    For example, with two reference images you might write a prompt like:
    "Place the character from image_1 into the landscape shown in image_2,
    matching the lighting and color palette of image_2."

    Args:
        prompt: Text description of the desired image(s). When reference images
            are provided, use image_1, image_2, … image_N in the prompt to
            refer to specific input images by their position in the list.
        image_urls: Optional list of reference image URLs (up to 14). Each URL
            becomes a named reference: the first URL is image_1, the second is
            image_2, and so on. The prompt should explicitly mention these
            references to control how each image influences the output.
        num_images: Number of images to generate (1-4).
        resolution: Output resolution. Choices: "1K", "2K", "4K" (4K costs 2x).
        aspect_ratio: Output aspect ratio. Choices: "auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16".
        enable_web_search: Use web search for current events or real-world references.
        seed: Random seed for reproducibility (0-2147483647).
    """
    _name = "nano_banana_pro"
    mode = "edit" if image_urls else "txt2img"
    t0 = time.time()
    try:
        endpoint = (
            "fal-ai/nano-banana-pro/edit" if image_urls else "fal-ai/nano-banana-pro"
        )

        fal_args: dict = {
            "prompt": prompt,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "output_format": "jpeg",
        }
        if resolution:
            fal_args["resolution"] = resolution
        if enable_web_search:
            fal_args["enable_web_search"] = True
        if seed >= 0:
            fal_args["seed"] = seed
        if image_urls:
            fal_args["image_urls"] = image_urls

        result = await _fal_subscribe(endpoint, fal_args, tool_name=_name, ctx=ctx)

        urls = _extract_urls(result)
        if not urls:
            err = "No image URLs in FAL response"
            log_error(_name, ValueError(err), t0)
            raise ToolError(err)

        log_progress(_name, f"Downloading {len(urls)} image(s) from FAL...")
        # Download all images concurrently
        tmp_paths = await asyncio.gather(
            *[_download_to_tmp(url, suffix=".jpg") for url in urls]
        )
        # Upload all to S3 concurrently
        uploaded = list(await asyncio.gather(
            *[upload_async(tmp) for tmp in tmp_paths]
        ))

        log_progress(_name, "Upload complete.")
        response = {
            "output_images": uploaded,
            "info": {
                "num_images": len(uploaded),
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "mode": mode,
            },
        }
        log_done(_name, t0, response)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))
