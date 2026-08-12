"""MCP tool wrappers that bridge MCP ↔ wzrd functions.

Each tool handles: resolve inputs (URL/path→local) → call wzrd → write outputs
into the active project folder (see `project.py`) and return local paths.
Inputs still accept URLs; outputs never leave this machine.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional
from urllib.parse import urlparse

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project
from .file_io import resolve_input_async
from ._log import log_call, log_progress, log_done, log_error, logged_tool
from .server import mcp, get_timeout


# ---------------------------------------------------------------------------
# Tool 1: subtract_background_frame
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("subtract_background_frame"))
@logged_tool
async def subtract_background_frame(
    generated_image: str,
    background_image: str,
    threshold: int = 10,
    ramp: int = 20,
    gamma: float = 0.85,
    blur_radius: float = 0.004,
    ctx: Optional[Context] = None,
) -> dict:
    """Remove static projection surface background from a generated image, extracting the foreground subject for additive projection mapping using luminance-based differences.

    Args:
        generated_image: URL or path to the generated image with subject.
        background_image: URL or path to the background reference (no subject).
        threshold: Luminance difference cutoff (0-255). Higher = less sensitive. Range 5-30.
        ramp: Soft transition width above threshold. Higher = softer edges.
        gamma: Gamma correction. Values < 1.0 brighten the output.
        blur_radius: Mask feathering as fraction of image size.
    """
    _name = "subtract_background_frame"
    t0 = time.time()
    try:
        from wzrd.subtract_frame import subtract_background_file

        log_progress(_name, "Resolving inputs...")
        gen_path, bg_path = await asyncio.gather(
            resolve_input_async(generated_image, suffix=".png"),
            resolve_input_async(background_image, suffix=".png"),
        )
        out_path = project.output_path(_name, ".webp", kind=project.GENERATED)

        log_progress(_name, "Running background subtraction...")
        _creature_img, info = await asyncio.to_thread(
            subtract_background_file,
            generated_path=gen_path,
            background_path=bg_path,
            output_path=out_path,
            threshold=threshold,
            ramp=ramp,
            gamma=gamma,
            blur_radius=blur_radius,
        )

        result = {
            "creature_image": out_path,
            "info": {
                "mask_coverage": info.get("mask_coverage"),
                "creature_max_brightness": info.get("creature_max_brightness"),
                "creature_mean_brightness": info.get("creature_mean_brightness"),
            },
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 2: subtract_background_video
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("subtract_background_video"))
@logged_tool
async def subtract_background_video(
    video: str,
    background_image: str,
    threshold: int = 10,
    gamma: float = 0.85,
    ramp: int = 20,
    blur_radius: float = 0.004,
    codec: str = "libx264",
    crf: int = 23,
    ctx: Optional[Context] = None,
) -> dict:
    """Remove background from an entire video frame-by-frame for projection mapping content.

    This is a slower operation. Processing time scales with video length and resolution.

    Args:
        video: URL or path to the input video.
        background_image: URL or path to the background reference image.
        threshold: Luminance difference cutoff (0-255).
        gamma: Gamma correction. Values < 1.0 brighten output.
        ramp: Soft transition width above threshold.
        blur_radius: Mask feathering as fraction of image size.
        codec: Video codec (e.g. libx264, h264_videotoolbox).
        crf: Video quality — lower is better (0-51).
    """
    _name = "subtract_background_video"
    t0 = time.time()
    try:
        from wzrd.subtract_video import subtract_background_video as _subtract_bg_video

        log_progress(_name, "Resolving inputs...")
        vid_path, bg_path = await asyncio.gather(
            resolve_input_async(video, suffix=".mp4"),
            resolve_input_async(background_image, suffix=".png"),
        )
        out_path = project.output_path(_name, ".mp4", kind=project.VIDEO)

        log_progress(_name, "Processing video frames...")
        info = await asyncio.to_thread(
            _subtract_bg_video,
            video_path=vid_path,
            background_path=bg_path,
            output_path=out_path,
            threshold=threshold,
            ramp=ramp,
            gamma=gamma,
            blur_radius=blur_radius,
            codec=codec,
            crf=crf,
        )

        result = {
            "output_video": out_path,
            "info": {
                "frames_processed": info.get("frames_processed"),
                "fps": info.get("fps"),
                "video_size": info.get("video_size"),
            },
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 3: detect_projection_surface
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("detect_projection_surface"))
@logged_tool
async def detect_projection_surface(
    image: str,
    margin: float = 0.01,
    target_aspect_ratio: str = "16:9",
    output_resolution: int = 1920,
    ctx: Optional[Context] = None,
) -> dict:
    """Detect and extract a projection surface from a night-time photo of projector light on a projection surface.

    Finds the illuminated quadrilateral (screen/wall) and returns a rectified, perspective-corrected crop.

    Args:
        image: URL or path to photo containing the projection surface.
        margin: Inward shrinkage fraction (0.0–0.1) to avoid edge artifacts.
        target_aspect_ratio: Force output aspect ratio, e.g. "16:9" or "4:3".
        output_resolution: Output width in pixels. None keeps original resolution.
    """
    _name = "detect_projection_surface"
    t0 = time.time()
    try:
        from wzrd.detect import detect_projection_area

        log_progress(_name, "Resolving input...")
        img_path = await resolve_input_async(image, suffix=".png")
        out_path = project.output_path(_name, ".webp", kind=project.SURFACES)

        log_progress(_name, "Detecting projection surface...")
        _cropped, info = await asyncio.to_thread(
            detect_projection_area,
            image_path=img_path,
            output_path=out_path,
            margin=margin,
            target_aspect_ratio=target_aspect_ratio,
            output_resolution=output_resolution,
        )

        result = {
            "cropped_image": out_path,
            "info": {
                "corners": info.get("corners"),
                "original_size": info.get("original_size"),
                "cropped_size": info.get("cropped_size"),
                "margin": info.get("margin"),
                "target_aspect_ratio": info.get("target_aspect_ratio"),
            },
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 4: align_images
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("align_images"))
@logged_tool
async def align_images(
    source_image: str,
    target_image: str,
    max_features: int = 2500,
    ctx: Optional[Context] = None,
) -> dict:
    """Align a source image to match a target image's perspective.

    Uses SIFT/AKAZE feature matching with template matching fallback and optional ECC sub-pixel refinement.

    Args:
        source_image: URL or path to the image to warp.
        target_image: URL or path to the reference image to align to.
        max_features: SIFT feature detection limit.
    """
    _name = "align_images"
    t0 = time.time()
    try:
        from wzrd.align import align_images_file

        log_progress(_name, "Resolving inputs...")
        src_path, tgt_path = await asyncio.gather(
            resolve_input_async(source_image, suffix=".png"),
            resolve_input_async(target_image, suffix=".png"),
        )
        out_path = project.output_path(_name, ".webp", kind=project.SURFACES)

        log_progress(_name, "Running feature matching & alignment...")
        _warped, info = await asyncio.to_thread(
            align_images_file,
            source_path=src_path,
            target_path=tgt_path,
            output_path=out_path,
            max_features=max_features,
        )

        result = {
            "aligned_image": out_path,
            "info": {
                "confidence": info.get("confidence"),
                "method": info.get("method"),
                "num_inliers": info.get("num_inliers"),
                "num_matches": info.get("num_matches"),
            },
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 5: darken_surface
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("darken_surface"))
@logged_tool
async def darken_surface(
    image: str,
    max_brightness: float = 0.25,
    detail_boost: float = 1.15,
    target_aspect: str = "16:9",
    base_resolution: int = 1920,
    alignment_aids: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Darken a surface photo for additive projection mapping.

    Uses gradient-weighted histogram equalization to preserve detail while reducing brightness
    so that the projector output appears natural on the surface.

    Args:
        image: URL or path to the surface image.
        max_brightness: Luminance budget ceiling (0.0–1.0). Lower = darker output.
        detail_boost: Detail amplification factor.
        target_aspect: Output aspect ratio, e.g. "16:9".
        base_resolution: Output width in pixels.
        alignment_aids: Whether to generate an alignment aid video to be used for aligning the projector with the surface (only needed once per VJ session).
    """
    _name = "darken_surface"
    t0 = time.time()
    try:
        from wzrd.darken import darken_image_file

        log_progress(_name, "Resolving input...")
        img_path = await resolve_input_async(image, suffix=".png")
        # Alignment aids are written next to `out_path` by the library, so a
        # project-local output path puts them in the project too.
        out_path = project.output_path(_name, ".webp", kind=project.SURFACES)

        log_progress(_name, "Running darken pipeline...")
        result = await asyncio.to_thread(
            darken_image_file,
            input_path=img_path,
            output_path=out_path,
            max_brightness=max_brightness,
            detail_boost=detail_boost,
            target_aspect=target_aspect,
            base_resolution=base_resolution,
            alignment_aids=alignment_aids,
        )

        pil_image = result["image"]
        await asyncio.to_thread(pil_image.save, out_path, quality=90)

        response: dict = {"darkened_image": out_path}
        if result.get("video"):
            response["alignment_video"] = str(result["video"])

        log_done(_name, t0, response)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 6: prepare_surface
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("prepare_surface"))
@logged_tool
async def prepare_surface(
    night_image: str,
    day_image: str = "",
    target_aspect: str = "16:9",
    max_brightness: float = 0.20,
    margin: float = 0.01,
    alignment_aids: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Full pipeline to prepare a projection surface from photos.

    If both day and night images are provided: detect surface → align → darken.
    If only night image: darken only.

    Args:
        night_image: URL or path to nighttime photo with projector on.
        day_image: URL or path to daytime photo of the same surface (triggers full pipeline).
        target_aspect: Output aspect ratio.
        max_brightness: Luminance ceiling (0.0–1.0).
        margin: Detection margin for surface extraction.
        alignment_aids: Whether to generate an alignment aid video to be used for aligning the projector with the surface (only needed once per VJ session).
    """
    _name = "prepare_surface"
    t0 = time.time()
    try:
        from wzrd.prepare_surface import prepare_surface as _prepare_surface

        log_progress(_name, "Resolving inputs...")
        coros = [resolve_input_async(night_image, suffix=".png")]
        if day_image != "":
            coros.append(resolve_input_async(day_image, suffix=".png"))
        resolved = await asyncio.gather(*coros)
        night_path = resolved[0]
        day_path = resolved[1] if len(resolved) > 1 else None
        out_path = project.output_path(_name, ".webp", kind=project.SURFACES)

        if day_path:
            log_progress(_name, "Running full pipeline: detect → align → darken...")
        else:
            log_progress(_name, "Running darken-only pipeline...")

        result = await asyncio.to_thread(
            _prepare_surface,
            night_image_path=night_path,
            day_image_path=day_path,
            output_path=out_path,
            margin=margin,
            max_brightness=max_brightness,
            target_aspect=target_aspect,
            alignment_aids=alignment_aids,
        )

        pil_image = result["image"]
        await asyncio.to_thread(pil_image.save, out_path, quality=90)

        response: dict = {"surface_image": out_path}
        if result.get("video"):
            response["alignment_video"] = str(result["video"])

        log_done(_name, t0, response)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 7: extract_color_regions
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("extract_color_regions"))
@logged_tool
async def extract_color_regions(
    image: str,
    max_colors: int = 8,
    min_area_fraction: float = 0.01,
    surface_image: str = "",
    output_min_size: int = 512,
    delta_e_threshold: float = 5.0,
    background_threshold: float = 15.0,
    merge_same_color: bool = True,
    standardize_output_aspect: bool = True,
    ctx: Optional[Context] = None,
) -> dict:
    """Segment an image into color-based regions (islands) using greedy clustering in CIELAB color space.

    Automatically discovers the number of distinct colors. Useful for splitting a surface into zones for independent content projection.

    Args:
        image: URL or path to the input image.
        max_colors: Upper cap on number of distinct colors to keep.
        min_area_fraction: Minimum region size as fraction of total image area.
        surface_image: Optional surface image to extract corresponding region crops from.
        output_min_size: Minimum output dimension in pixels for each region crop.
        delta_e_threshold: CIELAB deltaE threshold for merging similar colors. Lower = more sensitive to subtle differences. Default 15.0.
        background_threshold: CIELAB deltaE from background below which pixels are treated as background. Default 15.0.
        merge_same_color: If true, all disconnected blobs of the same color are combined into one region. Default true.
        standardize_output_aspect: If true, pad output crops to the closest standard aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9). Masks are padded with black; surface crops are extended from source pixels. Default true.
    """
    _name = "extract_color_regions"
    t0 = time.time()
    try:
        from wzrd.islands import extract_color_regions as _extract_color_regions

        log_progress(_name, "Resolving inputs...")
        coros = [resolve_input_async(image, suffix=".png")]
        if surface_image != "":
            coros.append(resolve_input_async(surface_image, suffix=".png"))
        resolved = await asyncio.gather(*coros)
        img_path = resolved[0]
        surface_path = resolved[1] if len(resolved) > 1 else None
        out_dir = project.output_dir(_name, kind=project.REGIONS)

        log_progress(_name, "Running CIELAB color clustering...")
        regions, output_dir = await asyncio.to_thread(
            _extract_color_regions,
            image=img_path,
            output_dir=out_dir,
            max_colors=max_colors,
            min_area_fraction=min_area_fraction,
            surface=surface_path,
            output_min_size=output_min_size,
            delta_e_threshold=delta_e_threshold,
            background_threshold=background_threshold,
            merge_same_color=merge_same_color,
            standardize_output_aspect=standardize_output_aspect,
        )

        from pathlib import Path

        out = Path(output_dir)

        # Every region file already sits in the project's regions/<run>/ dir —
        # just report where each one landed.
        published_regions = []
        for r in regions:
            entry = {"source_box": r.get("source_box")}
            if "standard_aspect" in r:
                entry["aspect_ratio"] = r["standard_aspect"]
            elif r.get("source_box"):
                w, h = r["source_box"]["width"], r["source_box"]["height"]
                entry["aspect_ratio"] = f"{w / h:.3f}"
            if "crop_filename" in r:
                entry["region_mask"] = str(out / r["crop_filename"])
            if "surface_filename" in r:
                entry["region_surface_image"] = str(out / r["surface_filename"])
            published_regions.append(entry)

        metadata_file = out / "islands.json"
        result = {
            "regions": published_regions,
            "regions_dir": str(out),
            "metadata_path": str(metadata_file) if metadata_file.exists() else None,
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 8: reproject_video
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("reproject_video"))
@logged_tool
async def reproject_video(
    video: str,
    x: int,
    y: int,
    width: int,
    height: int,
    target_aspect: str = "16:9",
    base_resolution: int = 1920,
    codec: str = "libx264",
    crf: int = 23,
    ctx: Optional[Context] = None,
) -> dict:
    """Place a processed video (typically a segmented section of a projection surface) back at its original position of the full projection canvas.

    Used to reposition generated island regions back into a full-frame projection output for additive compositing.
    The x, y, width, height values come from the source_box returned by extract_color_regions for the specific region being reprojected.

    Args:
        video: URL or path to the island video.
        x: Island x-offset on canvas (from source_box).
        y: Island y-offset on canvas (from source_box).
        width: Island source width in pixels (from source_box).
        height: Island source height in pixels (from source_box).
        target_aspect: Canvas aspect ratio, e.g. "16:9".
        base_resolution: Canvas width in pixels.
        codec: Video codec.
        crf: Quality setting — lower is better (0-51).
    """
    _name = "reproject_video"
    t0 = time.time()
    try:
        from wzrd.reproject import reproject_video_with_aspect

        log_progress(_name, "Resolving inputs...")
        vid_path = await resolve_input_async(video, suffix=".mp4")
        out_path = project.output_path(_name, ".mp4", kind=project.VIDEO)

        log_progress(_name, "Reprojecting video onto canvas...")
        info = await asyncio.to_thread(
            reproject_video_with_aspect,
            video_path=vid_path,
            island_metadata={"x": x, "y": y, "width": width, "height": height},
            output_path=out_path,
            target_aspect=target_aspect,
            base_resolution=base_resolution,
            crf=crf,
            codec=codec,
        )

        result = {
            "output_video": out_path,
            "info": {
                "frames_processed": info.get("frames_processed"),
                "fps": info.get("fps"),
                "canvas_size": info.get("canvas_size"),
                "island_position": info.get("island_position"),
            },
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 9: texture_flow
#
# THE LAST EDEN DEPENDENCY — disabled in tools_config.json until it is rehomed.
# It calls a ComfyUI deployment (`comfyui-wzrd-STAGE` / `ComfyUIPremium`) that
# lives in the *edenartlab* Modal workspace, feeds it public input URLs, and
# reads outputs back as S3 keys in Eden's bucket. Nothing here can be fixed by
# a local refactor; the workflow + deploy script have to move first.
# See docs/TODO/eden-decoupling.md.
# ---------------------------------------------------------------------------

# S3/CloudFront settings for downloading TextureFlow outputs. No Eden default:
# a rehomed deployment must name its own bucket.
_TF_BUCKET = os.getenv("AWS_BUCKET_NAME")
_TF_REGION = os.getenv("AWS_REGION_NAME", "us-east-1")
_TF_CLOUDFRONT = os.getenv("CLOUDFRONT_URL")


def _tf_file_url(filename: str) -> str:
    """Build a download URL for an S3 key from the TextureFlow result."""
    if _TF_CLOUDFRONT:
        return f"{_TF_CLOUDFRONT}/{filename}"
    if not _TF_BUCKET:
        raise ToolError(
            "TextureFlow returned an S3 key but no bucket is configured. "
            "Set AWS_BUCKET_NAME (or CLOUDFRONT_URL) for the deployment that "
            "produced it — see docs/TODO/eden-decoupling.md."
        )
    return f"https://{_TF_BUCKET}.s3.{_TF_REGION}.amazonaws.com/{filename}"


def _tf_extract_urls(obj):
    """Recursively extract download URLs from a ComfyUI upload_result structure."""
    urls = []
    if isinstance(obj, str) and obj.startswith("http"):
        local_name = os.path.basename(urlparse(obj).path) or "output"
        urls.append((obj, local_name))
    elif isinstance(obj, dict):
        if "filename" in obj and isinstance(obj["filename"], str):
            s3_key = obj["filename"]
            urls.append((_tf_file_url(s3_key), os.path.basename(s3_key)))
        else:
            for v in obj.values():
                urls.extend(_tf_extract_urls(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            urls.extend(_tf_extract_urls(item))
    return urls


def _tf_extract_first_url(obj) -> str | None:
    """Extract the first download URL from a ComfyUI output node value."""
    urls = _tf_extract_urls(obj)
    return urls[0][0] if urls else None


@mcp.tool(timeout=get_timeout("texture_flow"))
@logged_tool
async def texture_flow(
    images: list[str],
    n_seconds: float = 5.0,
    width: int = 512,
    height: int = 384,
    base_model: str = "SD15/juggernaut_reborn.safetensors",
    use_controlnet1: bool = False,
    control_input: str = "",
    diffusion_mask: str = "",
    preprocessor1: str = "Scribble_XDoG_Preprocessor",
    controlnet_strength1: float = 0.45,
    denoise: float = 1.0,
    control_input_fit_strategy: str = "fill / crop",
    map_shape_input_to_ip_masks: bool = False,
    mapping_mode: str = "concentric_circles_outwards",
    n_steps: int = 6,
    motion_scale: float = 1.15,
    use_upscale: bool = False,
    upscale_resolution: int = 1024,
    upscale_esrgan: bool = False,
    seed: int = -1,
    ctx: Optional[Context] = None,
) -> dict:
    """Generate a smooth, morphing animation video from style images using the TextureFlow AI model (runs on a remote Modal GPU endpoint).

    TextureFlow creates trippy, artistic, morphing animations from 1-6 style images based on AnimateDiff, SDv15 and IP-adapter and controlnet.
    Ideal for VJ loops, animated logos, abstract animations, and projection mapping content.

    The style images drive the textures, content, and colors of the output video, but style images will never appear in the generated video exactly as keyframes.
    Scale video length proportionally with number of style images (~3s per image is a good rule of thumb).

    Optional shape guidance (controlnet):
    - Set use_controlnet1=true and provide a control_input image/video to add shape guidance.
    - Great for embedding logo contours, projection surface lines, or driving animations with simple motion videos.
    - Without control_input, TextureFlow often produces visually appealing results out-of-the-box.
    - With control_input, results can be amazing but may require tuning (especially controlnet_strength1).

    Example use cases:
    - Abstract artistic animation from style image(s) (will deform input, great for smooth animated textures)
    - Mixing multiple style images with various mapping modes for VJ content
    - Logo animation: logo as control_input + texture images as style
    - Motion-driven animation: simple motion video/GIF as control_input
    - Animated QR codes: QR code as control_input with preprocessor1="none"

    Args:
        images: 1-6 style image URLs driving the content/textures/colors of the video. More images = more variety.
        n_seconds: Video length in seconds (2.0-24.0). Scale with number of style images (~3s per image).
        width: Video width in pixels (320-1280, step 32). High values may cause duplication artifacts and are very slow.
        height: Video height in pixels (320-1280, step 32). High values may cause duplication artifacts and are very slow.
        base_model: SD1.5 checkpoint. Options: "SD15/juggernaut_reborn.safetensors" (realistic, best default), "SD15/darkSushiMixMix_225D.safetensors" (more creative/abstract, softer, pastel colors), "SD15/protogenV22Anime_protogenV22.safetensors" (anime/cartoon, flat colors).
        use_controlnet1: Enable controlnet shape guidance. Requires control_input.
        control_input: URL to an image/video for shape guidance. Used as controlnet input when use_controlnet1=true. Can be a logo, surface photo, motion video, or QR code.
        diffusion_mask: Optional image/video mask URL controlling which regions are affected by diffusion. White regions will get animated, black regions remain black. Use this to generated island videos. Diffusion_mask can also be eg a shape filling video.
        preprocessor1: Shape guidance type (only used when use_controlnet1=true). Options: "Scribble_XDoG_Preprocessor" (rough shape, good quality), "CannyEdgePreprocessor" (strong detailed edges), "DepthAnythingV2Preprocessor" (depth structure, ignores edges), "AnyLineArtPreprocessor_aux" (lineart), "DensePosePreprocessor" (human pose extraction), "none" (luminance/QR mode - maintains dark/bright regions).
        controlnet_strength1: Shape guidance strength 0.0-1.0 (only when use_controlnet1=true). Default 0.45. Subtle: 0.35-0.45, strong: 0.45-0.65. Too high can look bad.
        denoise: AI strength on top of shape input 0.1-1.0 (only when use_controlnet1=true). Lower values (0.8-0.9) preserve some of the control input's colors/shape. This can be used to provide eg a logo / rough color outline and then diffuse an animation on top of that. Can be used with or without controlnet_strength1.
        control_input_fit_strategy: How to resize shape input to target dimensions. "fill / crop" (recommended), "stretch" (distorts), "pad" (can cause artifacts).
        map_shape_input_to_ip_masks: Advanced: use color clusters from shape input to spatially map style images. Only enable when shape input has flat single-color regions. Overrides mapping_mode.
        mapping_mode: Motion pattern for style image mapping. Options: "concentric_circles_inwards", "concentric_circles_outwards" (good default), "concentric_rectangles_inwards", "concentric_rectangles_outwards", "rotating_segments_clockwise", "rotating_segments_counter_clockwise" (avoid unless asked), "pushing_segments_clockwise", "pushing_segments_counter_clockwise", "vertical_stripes_left", "vertical_stripes_right", "horizontal_stripes_up" (good for demos), "horizontal_stripes_down".
        n_steps: LCM denoising steps (4-14). Lower = faster/cheaper, higher = slightly better quality. Runtime scales linearly with n_steps.
        motion_scale: Animation motion strength (0.7-1.4). Default 1.1 is good. 0.9 = subtle, 1.25 = more motion. Above 1.3 is rarely desirable.
        use_upscale: Enable HD upscaling second pass. Disable for faster/cheaper experimentation, enable for HD videos.
        upscale_resolution: Max dimension for latent upscale (1024-1536, step 64). Only used when use_upscale=true.
        upscale_esrgan: Enable ESRGAN postprocessing for full HD (1080p). Great for sharp/realistic, less ideal for organic/stylistic. Only used when use_upscale=true.
        seed: Random seed for reproducibility (0-2147483647). Leave as None for random, set manually when doing param gridsearches (rarely needed).
    """
    _name = "texture_flow"
    t0 = time.time()
    try:
        import modal

        args = {
            "images": images,
            "n_seconds": n_seconds,
            "width": width,
            "height": height,
            "base_model": base_model,
            "use_controlnet1": use_controlnet1,
            "preprocessor1": preprocessor1,
            "controlnet_strength1": controlnet_strength1,
            "denoise": denoise,
            "control_input_fit_strategy": control_input_fit_strategy,
            "map_shape_input_to_ip_masks": map_shape_input_to_ip_masks,
            "mapping_mode": mapping_mode,
            "n_steps": n_steps,
            "motion_scale": motion_scale,
            "use_upscale": use_upscale,
            "upscale_resolution": upscale_resolution,
            "upscale_esrgan": upscale_esrgan,
        }

        if control_input != "":
            args["control_input"] = control_input
        if diffusion_mask != "":
            args["diffusion_mask"] = diffusion_mask
        if seed >= 0:
            args["seed"] = seed

        app_name = os.getenv("MODAL_APP_NAME", "comfyui-wzrd-STAGE")
        cls_name = os.getenv("MODAL_CLS_NAME", "ComfyUIPremium")

        log_progress(_name, f"Calling Modal endpoint ({app_name}/{cls_name})...")

        cls = modal.Cls.from_name(app_name, cls_name)
        instance = cls()
        result = await instance.run.remote.aio(tool_key="texture_flow", args=args)

        log_progress(_name, "Extracting output URLs...")

        # Map ComfyUI intermediate output keys to descriptive names
        _INTERMEDIATE_KEY_MAP = {
            "pass_1": "first_pass_video",
            "control_signal_1": "control_signal_preview",
            "mapping_motion": "mapping_motion_preview",
        }

        # Extract main output video
        response: dict = {}
        if isinstance(result, dict):
            # Main output: top-level "output" key from ComfyUI handler
            main_output = result.get("output")
            if main_output:
                url = _tf_extract_first_url(main_output)
                if url:
                    response["output_video"] = url

            # Intermediate/debug outputs
            intermediates = result.get("intermediate_outputs", {})
            if isinstance(intermediates, dict):
                for raw_key, named_key in _INTERMEDIATE_KEY_MAP.items():
                    node_val = intermediates.get(raw_key)
                    if node_val:
                        url = _tf_extract_first_url(node_val)
                        if url:
                            response[named_key] = url

        # Fallback: if we couldn't parse structured output, extract all URLs
        if "output_video" not in response:
            all_urls = _tf_extract_urls(result)
            if all_urls:
                response["output_video"] = all_urls[0][0]
                # Assign remaining URLs to intermediate keys in order
                fallback_keys = ["first_pass_video", "control_signal_preview", "mapping_motion_preview"]
                for i, (url, _) in enumerate(all_urls[1:]):
                    key = fallback_keys[i] if i < len(fallback_keys) else f"extra_output_{i}"
                    response[key] = url

        # Bring every remote result home before returning it.
        log_progress(_name, f"Downloading {len(response)} output(s)...")
        keys = list(response)
        paths = await asyncio.gather(*(
            project.download_async(response[k], tool=f"{_name}_{k}", kind=project.GENERATED)
            for k in keys
        ))
        response = dict(zip(keys, paths))

        log_done(_name, t0, response)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool: build_layerpack
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("build_layerpack"))
@logged_tool
async def build_layerpack(
    masks: list[str],
    surface_image: str = "",
    mask_ids: list[str] | None = None,
    tags: dict | None = None,
    projector_width: int = 0,
    projector_height: int = 0,
    source_capture: str = "",
    canny: str = "",
    feather_px: int = 1,
    include_background: bool = False,
    pack_name: str = "",
    ctx: Optional[Context] = None,
) -> dict:
    """Build a WZRD layer pack (pack.json + masks/ + references/) for the realtime render-core.

    Takes a set of semantic mask images (typically the output of `extract_color_regions`,
    possibly hand-edited / SAM-refined) and an optional per-mask tags structure, and
    emits a self-contained layer pack the realtime engine consumes.

    Per-layer identity (id, tags, group, parent) is stable across re-shoots and
    re-segmentations — bindings target ids, not mask filenames.

    Args:
        masks: List of mask image URLs/paths. Each becomes one layer slice
            (R8 grayscale, projector-resolution). Hard cap: 256 layers.
        surface_image: URL/path to the darkened/aligned surface image used as
            preview overlay and to derive projector_resolution if not given.
        mask_ids: Optional parallel list of semantic ids (length must match `masks`).
            Required if you want stable ids; otherwise ids default to mask stems.
        tags: Optional structured tags dict: {
            "layers": {
                "<mask_filename_or_id>": {
                    "label": "...", "tags": ["..."], "parent": "...", "z": 1
                }
            },
            "groups": [{"id": "...", "members": ["..."]}]
        }
        projector_width: Width in px. Defaults to surface image width.
        projector_height: Height in px. Defaults to surface image height.
        source_capture: Optional original photo URL/path (copied into references/).
        canny: Optional canny/edge aid URL/path (copied into references/).
        feather_px: Mask edge feathering radius (default 1; 0 = hard edges).
        include_background: If true, auto-generates `000_background.png` as the
            complement of all input masks.
        pack_name: Folder name for the pack inside the project's `packs/`
            directory. Defaults to a timestamped name.
    """
    _name = "build_layerpack"
    t0 = time.time()
    try:
        from wzrd.layerpack import build_layerpack as _build

        log_progress(_name, "Resolving inputs...")
        mask_coros = [resolve_input_async(m, suffix=".png") for m in masks]
        extra_coros: list = []
        if surface_image:
            extra_coros.append(resolve_input_async(surface_image, suffix=".png"))
        if source_capture:
            extra_coros.append(resolve_input_async(source_capture, suffix=".png"))
        if canny:
            extra_coros.append(resolve_input_async(canny, suffix=".png"))
        resolved_masks = await asyncio.gather(*mask_coros)
        resolved_extras = await asyncio.gather(*extra_coros)

        extras_iter = iter(resolved_extras)
        surf_path = next(extras_iter) if surface_image else None
        src_path = next(extras_iter) if source_capture else None
        cny_path = next(extras_iter) if canny else None

        if mask_ids and len(mask_ids) != len(resolved_masks):
            raise ToolError("mask_ids length must match masks length")

        # Stage masks in a temp dir with stable, ordered filenames so the
        # layerpack discovery sees them in the order the caller supplied.
        from pathlib import Path
        from tempfile import mkdtemp
        masks_dir = Path(mkdtemp(prefix="wzrd_masks_")) / "masks_in"
        masks_dir.mkdir()
        from shutil import copyfile
        ordered_filenames: list[str] = []
        for i, src in enumerate(resolved_masks):
            stem = mask_ids[i] if mask_ids else f"layer_{i:03d}"
            dest_name = f"{i:03d}_{stem}.png"
            copyfile(src, masks_dir / dest_name)
            ordered_filenames.append(dest_name)

        # If caller supplied per-mask metadata under tags["layers"], rekey it to
        # match our deterministic staging filenames (so it actually applies).
        rekeyed_tags = dict(tags or {})
        if mask_ids and rekeyed_tags.get("layers"):
            src_layers = rekeyed_tags["layers"]
            new_layers: dict = {}
            for i, stem in enumerate(mask_ids):
                if stem in src_layers:
                    entry = dict(src_layers[stem])
                    entry.setdefault("id", stem)
                    new_layers[ordered_filenames[i]] = entry
                else:
                    new_layers[ordered_filenames[i]] = {"id": stem}
            rekeyed_tags["layers"] = new_layers
        elif mask_ids:
            rekeyed_tags["layers"] = {
                ordered_filenames[i]: {"id": mask_ids[i]}
                for i in range(len(mask_ids))
            }

        projector_resolution = None
        if projector_width > 0 and projector_height > 0:
            projector_resolution = (projector_width, projector_height)

        # The pack lands in the project so render-core can load it directly:
        # a scene.json at the project root points at `packs/<name>` and the
        # engine resolves masks relative to pack.json, off local disk.
        pack_dir = Path(project.output_dir(_name, kind=project.PACKS, name=pack_name))

        log_progress(_name, "Building layer pack...")
        scene = await asyncio.to_thread(
            _build,
            masks_dir=str(masks_dir),
            output_dir=str(pack_dir),
            surface=surf_path,
            tags=rekeyed_tags,
            projector_resolution=projector_resolution,
            source_capture=src_path,
            canny=cny_path,
            feather_px=feather_px,
            include_background=include_background,
        )

        # `_build` already wrote pack.json with mask paths relative to the pack
        # dir — exactly what render-core expects. Nothing to rewrite.
        result = {
            "scene": scene,
            "pack_dir": str(pack_dir),
            "pack_json": str(pack_dir / "pack.json"),
            "layer_count": len(scene["layers"]),
            "projector_resolution": scene["projector_resolution"],
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 10: simulate_view
# ---------------------------------------------------------------------------
@mcp.tool(timeout=get_timeout("simulate_view"))
@logged_tool
async def simulate_view(
    projector_video: str,
    surface_image: str,
    surface_weight: float = 0.35,
    projection_strength: float = 1.0,
    ambient: float = 0.0,
    codec: str = "libx264",
    crf: int = 18,
    ctx: Optional[Context] = None,
) -> dict:
    """Simulate the real-world viewing experience of additive projection mapping.

    Composites a projector output video (purely additive light on black background)
    onto a projection surface image to predict what a human viewer would actually see.
    Non-reflective regions (sky, etc. turned to black) stay dark — projected light
    is modulated by surface reflectivity derived from the surface image luminance.

    Use this after generating projector content (via texture_flow, kling, or
    subtract_background_video) to preview how the projection will look on the
    actual surface before projecting.

    Args:
        projector_video: URL or path to the projector output video (additive light
            content — a background-subtracted kling video or raw textureflow output).
        surface_image: URL or path to the projection surface image. Ideally
            pre-processed so non-reflective regions (sky, etc.) are pure black
            (e.g. via nano_banana_pro masking).
        surface_weight: How visible the underlying surface texture is (0.0-1.0).
            Lower = more contrast from projection only, higher = more visible
            ambient surface detail. Default 0.35.
        projection_strength: Multiplier for projected light intensity (0.5-2.0).
            Values > 1.0 simulate a brighter projector. Default 1.0.
        ambient: Minimum reflectivity floor for non-black surface pixels (0.0-0.3).
            Adds subtle light bounce on dark (but not black) surface areas.
        codec: Video codec (e.g. libx264, h264_videotoolbox).
        crf: Video quality — lower is better (0-51).
    """
    _name = "simulate_view"
    t0 = time.time()
    try:
        from wzrd.simulate_view import simulate_view as _simulate_view

        log_progress(_name, "Resolving inputs...")
        vid_path, surf_path = await asyncio.gather(
            resolve_input_async(projector_video, suffix=".mp4"),
            resolve_input_async(surface_image, suffix=".png"),
        )
        out_path = project.output_path(_name, ".mp4", kind=project.VIDEO)

        log_progress(_name, "Compositing projector video onto surface...")
        info = await asyncio.to_thread(
            _simulate_view,
            projector_video_path=vid_path,
            surface_image_path=surf_path,
            output_path=out_path,
            surface_weight=surface_weight,
            projection_strength=projection_strength,
            ambient=ambient,
            codec=codec,
            crf=crf,
        )

        result = {
            "simulated_view_video": out_path,
            "info": {
                "frames_processed": info.get("frames_processed"),
                "fps": info.get("fps"),
                "video_size": info.get("video_size"),
                "surface_weight": info.get("surface_weight"),
                "projection_strength": info.get("projection_strength"),
            },
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))
