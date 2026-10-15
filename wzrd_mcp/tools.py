"""MCP tool wrappers that bridge MCP ↔ wzrd functions.

Each tool: resolve inputs (URL/path → local) → call wzrd → write outputs into
the active project folder (see `project.py`) → return local paths. Logging,
error wrapping and the text-only result come from `server.tool()`.

Descriptions carry the contract (arguments, ranges, traps); the craft — which
tool when, recipes, prompt formulas — lives in the skills (`skills/`).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from shutil import copyfile
from tempfile import mkdtemp
from typing import Annotated, Any, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from . import jobs, library, project, provenance
from ._keepalive import with_keepalive
from ._log import log_progress
from .file_io import resolve_input_async
from .media_tools import attach_clip
from .server import tool


# ---------------------------------------------------------------------------
# The locked project canvas (C1, D7)
# ---------------------------------------------------------------------------
def _canvas(**sizing: Any) -> Optional[tuple[int, int]]:
    """The active project's canvas, refusing explicit sizing args that would fight it."""
    size = project.canvas()
    if size:
        given = sorted(k for k, v in sizing.items() if v not in (None, 0))
        if given:
            raise ToolError(
                f"project {project.active_project()!r} has a locked canvas {size[0]}x{size[1]}; "
                f"omit {', '.join(given)} — outputs are conformed to the canvas"
            )
    return size


def _check_canvas_image(path: str, size: tuple[int, int], what: str) -> None:
    from PIL import Image

    with Image.open(path) as im:
        got = im.size
    if got != tuple(size):
        raise ToolError(
            f"{what} is {got[0]}x{got[1]}, not the project canvas {size[0]}x{size[1]} — "
            f"make it in this project (prepare_surface / darken_surface conform to the canvas)"
        )


def _conform_saved(image: Any, size: Optional[tuple[int, int]]) -> Any:
    """Belt and braces: a surface tool's PIL output, cover-cropped to the canvas if it isn't."""
    if size and image.size != tuple(size):
        from wzrd.utils import conform_to_size

        return conform_to_size(image, size)
    return image


# ---------------------------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------------------------
@tool(title="Subtract background (image)", tags={"content"}, project_arg=True)
async def subtract_background_frame(
    generated_image: str,
    background_image: str,
    threshold: int = 10,
    ramp: int = 20,
    gamma: float = 0.85,
    blur_radius: float = 0.004,
    ctx: Optional[Context] = None,
) -> dict:
    """Keep only what a generated image ADDED on top of the static surface photo, on black — the projector-ready still.

    Args:
        generated_image: Path or URL of the generated image (surface + subject).
        background_image: Path or URL of the bare surface it was generated on.
        threshold: Luminance difference cutoff (0-255, useful 5-30). Higher = less sensitive.
        ramp: Soft transition width above threshold. Higher = softer edges.
        gamma: Gamma correction. Values < 1.0 brighten the output.
        blur_radius: Mask feathering as fraction of image size.
    """
    from wzrd.subtract_frame import subtract_background_file

    gen_path, bg_path = await asyncio.gather(
        resolve_input_async(generated_image, suffix=".png"),
        resolve_input_async(background_image, suffix=".png"),
    )
    out_path = project.output_path("subtract_background_frame", ".webp", kind=project.GENERATED)
    _img, info = await asyncio.to_thread(
        subtract_background_file,
        generated_path=gen_path, background_path=bg_path, output_path=out_path,
        threshold=threshold, ramp=ramp, gamma=gamma, blur_radius=blur_radius,
    )
    return {
        "creature_image": out_path,
        "info": {k: info.get(k) for k in ("mask_coverage", "creature_max_brightness", "creature_mean_brightness")},
    }


@tool(title="Subtract background (video)", tags={"content"}, project_arg=True)
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
    """Turn a generated video of the surface into projector output: per frame, keep only what differs from the static surface photo; everything else goes black.

    Camera drift or global flicker in the source shows up here as noise across
    the whole frame — inspect the result (view_media) before using it.
    Slow-ish: scales with length and resolution.

    Args:
        video: Path or URL of the generated video (surface + animation).
        background_image: Path or URL of the bare surface (the video's start frame or the surface photo).
        threshold: Luminance difference cutoff (0-255). Raise it if the result is noisy.
        gamma: Gamma correction. Values < 1.0 brighten output.
        ramp: Soft transition width above threshold.
        blur_radius: Mask feathering as fraction of image size.
        codec: ffmpeg video codec (libx264 everywhere; h264_videotoolbox on a Mac is faster).
        crf: Video quality — lower is better (0-51).
    """
    from wzrd.subtract_video import subtract_background_video as _subtract_bg_video

    vid_path, bg_path = await asyncio.gather(
        resolve_input_async(video, suffix=".mp4"),
        resolve_input_async(background_image, suffix=".png"),
    )
    out_path = project.output_path("subtract_background_video", ".mp4", kind=project.VIDEO)
    log_progress("subtract_background_video", "Processing video frames...")
    info = await asyncio.to_thread(
        _subtract_bg_video,
        video_path=vid_path, background_path=bg_path, output_path=out_path,
        threshold=threshold, ramp=ramp, gamma=gamma, blur_radius=blur_radius, codec=codec, crf=crf,
    )
    return {
        "output_video": out_path,
        "info": {k: info.get(k) for k in ("frames_processed", "fps", "video_size")},
    }


# ---------------------------------------------------------------------------
# Surface preparation
# ---------------------------------------------------------------------------
@tool(title="Detect projection surface", tags={"surface"}, project_arg=True)
async def detect_projection_surface(
    image: str,
    margin: float = 0.01,
    target_aspect_ratio: Optional[str] = None,
    output_resolution: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Find the lit quadrilateral in a night photo of projector light on a surface and return a perspective-corrected crop.

    In a project with a canvas the whole lit quad is rectified to exactly the
    canvas (it is the projector raster) — leave the two sizing args unset.

    Args:
        image: Path or URL of the photo containing the projection surface.
        margin: Inward shrinkage fraction (0.0-0.1) to avoid edge artifacts.
        target_aspect_ratio: Projects without a canvas only: force output aspect ratio (default "16:9").
        output_resolution: Projects without a canvas only: output width in pixels (default 1920).
    """
    from wzrd.detect import detect_projection_area

    size = _canvas(target_aspect_ratio=target_aspect_ratio, output_resolution=output_resolution)
    sizing = {"output_size": size} if size else {
        "target_aspect_ratio": target_aspect_ratio or "16:9",
        "output_resolution": output_resolution or 1920,
    }
    img_path = await resolve_input_async(image, suffix=".png")
    out_path = project.output_path("detect_projection_surface", ".webp", kind=project.SURFACES)
    _cropped, info = await asyncio.to_thread(
        detect_projection_area,
        image_path=img_path, output_path=out_path, margin=margin, **sizing,
    )
    return {
        "cropped_image": out_path,
        "info": {k: info.get(k) for k in ("corners", "original_size", "cropped_size", "margin", "target_aspect_ratio")},
    }


@tool(title="Align images", tags={"surface"}, project_arg=True)
async def align_images(
    source_image: str,
    target_image: str,
    max_features: int = 2500,
    ctx: Optional[Context] = None,
) -> dict:
    """Warp a source image onto a target image's perspective (SIFT/AKAZE + RANSAC, template-matching fallback, ECC refinement).

    Args:
        source_image: Path or URL of the image to warp.
        target_image: Path or URL of the reference image to align to.
        max_features: SIFT feature detection limit.
    """
    from wzrd.align import align_images_file

    src_path, tgt_path = await asyncio.gather(
        resolve_input_async(source_image, suffix=".png"),
        resolve_input_async(target_image, suffix=".png"),
    )
    out_path = project.output_path("align_images", ".webp", kind=project.SURFACES)
    _warped, info = await asyncio.to_thread(
        align_images_file,
        source_path=src_path, target_path=tgt_path, output_path=out_path, max_features=max_features,
    )
    return {
        "aligned_image": out_path,
        "info": {k: info.get(k) for k in ("confidence", "method", "num_inliers", "num_matches")},
    }


@tool(title="Darken surface", tags={"surface"}, project_arg=True)
async def darken_surface(
    image: str,
    max_brightness: float = 0.25,
    detail_boost: float = 1.15,
    target_aspect: Optional[str] = None,
    base_resolution: Optional[int] = None,
    alignment_aids: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Darken a surface photo (gradient-weighted histogram equalization in CIELAB) so it looks like the surface at night, detail kept — the base every generation is drawn on.

    In a project with a canvas the photo is cover-cropped and sized to exactly
    the canvas — leave the two sizing args unset.

    Args:
        image: Path or URL of the surface image.
        max_brightness: Luminance ceiling (0.0-1.0). Lower = darker output.
        detail_boost: Detail amplification factor.
        target_aspect: Projects without a canvas only: output aspect ratio (default "16:9").
        base_resolution: Projects without a canvas only: output width in pixels (default 1920).
        alignment_aids: Also render an alignment-aid video for lining the projector up with the surface (once per session).
    """
    from wzrd.darken import darken_image_file

    size = _canvas(target_aspect=target_aspect, base_resolution=base_resolution)
    img_path = await resolve_input_async(image, suffix=".png")
    # Alignment aids are written next to `out_path` by the library.
    out_path = project.output_path("darken_surface", ".webp", kind=project.SURFACES)
    result = await asyncio.to_thread(
        darken_image_file,
        input_path=img_path, output_path=out_path, max_brightness=max_brightness,
        detail_boost=detail_boost,
        target_aspect=f"{size[0]}x{size[1]}" if size else target_aspect or "16:9",
        base_resolution=max(size) if size else base_resolution or 1920,
        alignment_aids=alignment_aids,
    )
    await asyncio.to_thread(_conform_saved(result["image"], size).save, out_path, quality=90)
    response: dict = {"darkened_image": out_path}
    if result.get("video"):
        response["alignment_video"] = str(result["video"])
    return response


@tool(title="Prepare surface", tags={"surface"}, project_arg=True)
async def prepare_surface(
    night_image: str,
    day_image: Optional[str] = None,
    target_aspect: Optional[str] = None,
    max_brightness: float = 0.20,
    margin: float = 0.01,
    alignment_aids: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Photos → the dark, projector-aligned surface image every later step builds on.

    With a day photo: detect the lit area in the night photo → align the day
    photo to it → darken. Night photo only: darken only. In a project with a
    canvas the output is exactly the canvas — leave target_aspect unset.

    Args:
        night_image: Path or URL of the night photo with the projector on (white frame).
        day_image: Path or URL of a daytime photo of the same surface (enables detect + align).
        target_aspect: Projects without a canvas only: output aspect ratio (default "16:9").
        max_brightness: Luminance ceiling (0.0-1.0).
        margin: Detection margin for surface extraction.
        alignment_aids: Also render an alignment-aid video for lining the projector up with the surface (once per session).
    """
    from wzrd.prepare_surface import prepare_surface as _prepare_surface

    size = _canvas(target_aspect=target_aspect)
    sizing = {
        "target_aspect": f"{size[0]}x{size[1]}", "base_resolution": max(size),
        "detect_output_size": size,
    } if size else {"target_aspect": target_aspect or "16:9"}
    coros = [resolve_input_async(night_image, suffix=".png")]
    if day_image:
        coros.append(resolve_input_async(day_image, suffix=".png"))
    resolved = await asyncio.gather(*coros)
    out_path = project.output_path("prepare_surface", ".webp", kind=project.SURFACES)
    log_progress("prepare_surface", "detect → align → darken" if day_image else "darken only")
    result = await asyncio.to_thread(
        _prepare_surface,
        night_image_path=resolved[0], day_image_path=resolved[1] if day_image else None,
        output_path=out_path, margin=margin, max_brightness=max_brightness,
        alignment_aids=alignment_aids, **sizing,
    )
    await asyncio.to_thread(_conform_saved(result["image"], size).save, out_path, quality=90)
    response: dict = {"surface_image": out_path}
    if result.get("video"):
        response["alignment_video"] = str(result["video"])
    return response


# ---------------------------------------------------------------------------
# Regions (islands)
# ---------------------------------------------------------------------------
@tool(title="Extract color regions", tags={"regions"}, project_arg=True)
async def extract_color_regions(
    image: str,
    max_colors: int = 8,
    min_area_fraction: float = 0.01,
    surface_image: Optional[str] = None,
    output_min_size: int = 512,
    delta_e_threshold: float = 5.0,
    background_threshold: float = 15.0,
    merge_same_color: bool = True,
    standardize_output_aspect: bool = True,
    ctx: Optional[Context] = None,
) -> dict:
    """Split a flat-color segmentation map into region masks ("islands"), each with its bounding box on the canvas.

    Input is usually a solid-color segmentation map of the surface (black =
    background). Per region: `region_mask` (full canvas — for
    build_layerpack), `region_crop_mask` + `region_surface_image` (cut to the
    region's box — the diffusion_mask / reference for content made for that
    region), `source_box`, `aspect_ratio`. A map at another size than the surface (or the project
    canvas) is uncropped into it — uniform scale, never stretched. Each
    region's `source_box` is its box on the canvas, padded to a Nano Banana
    native aspect: render that region's keyframes/videos from its
    region_surface_image at that aspect and they fit the box exactly.

    Args:
        image: Path or URL of the segmentation map.
        max_colors: Upper cap on distinct colors kept.
        min_area_fraction: Minimum region size as fraction of the image.
        surface_image: Optional surface image to also cut matching per-region surface crops from.
        output_min_size: Minimum output dimension in pixels for each crop.
        delta_e_threshold: CIELAB deltaE for merging similar colors. Lower = splits subtler differences.
        background_threshold: CIELAB deltaE from the background below which pixels count as background.
        merge_same_color: Combine disconnected blobs of the same color into one region.
        standardize_output_aspect: Pad crops to the nearest standard aspect (1:1, 3:2, 4:3, 16:9, 21:9, …; masks pad black, surface crops extend from source pixels).
    """
    from wzrd.islands import extract_color_regions as _extract_color_regions

    size = project.canvas()
    coros = [resolve_input_async(image, suffix=".png")]
    if surface_image:
        coros.append(resolve_input_async(surface_image, suffix=".png"))
    resolved = await asyncio.gather(*coros)
    if size and surface_image:
        _check_canvas_image(resolved[1], size, "surface_image")
    out_dir = project.output_dir("extract_color_regions", kind=project.REGIONS)
    regions, output_dir = await asyncio.to_thread(
        _extract_color_regions,
        image=resolved[0], output_dir=out_dir, max_colors=max_colors,
        min_area_fraction=min_area_fraction, surface=resolved[1] if surface_image else None,
        output_min_size=output_min_size, delta_e_threshold=delta_e_threshold,
        background_threshold=background_threshold, merge_same_color=merge_same_color,
        standardize_output_aspect=standardize_output_aspect, frame_size=size,
    )
    out = Path(output_dir)
    published = []
    for r in regions:
        entry: dict[str, Any] = {"source_box": r.get("source_box")}
        if "standard_aspect" in r:
            entry["aspect_ratio"] = r["standard_aspect"]
        elif r.get("source_box"):
            entry["aspect_ratio"] = f"{r['source_box']['width'] / r['source_box']['height']:.3f}"
        # region_mask = the full-canvas mask (build_layerpack); region_crop_mask =
        # the same region cut to its source_box (texture_flow diffusion_mask).
        entry["region_mask"] = str(out / r["region_mask"])
        if "crop_filename" in r:
            entry["region_crop_mask"] = str(out / r["crop_filename"])
        if "surface_filename" in r:
            entry["region_surface_image"] = str(out / r["surface_filename"])
        published.append(entry)
    metadata_file = out / "islands.json"
    return {
        "regions": published,
        "regions_dir": str(out),
        "metadata_path": str(metadata_file) if metadata_file.exists() else None,
    }


@tool(title="Reproject region video", tags={"regions"}, project_arg=True)
async def reproject_video(
    video: str,
    x: int,
    y: int,
    width: int,
    height: int,
    target_aspect: Optional[str] = None,
    base_resolution: Optional[int] = None,
    codec: str = "libx264",
    crf: int = 23,
    ctx: Optional[Context] = None,
) -> dict:
    """Place a region ("island") video back at its position on the full projection canvas, black elsewhere.

    x, y, width, height are the region's `source_box` from extract_color_regions.
    A video of another aspect than the box is fit uniformly, centred — never
    stretched. In a project with a canvas the output is the canvas size.

    Args:
        video: Path or URL of the island video.
        x: source_box.x
        y: source_box.y
        width: source_box.width
        height: source_box.height
        target_aspect: Projects without a canvas only: canvas aspect ratio (default "16:9").
        base_resolution: Projects without a canvas only: canvas width in pixels (default 1920).
        codec: ffmpeg video codec.
        crf: Quality — lower is better (0-51).
    """
    from wzrd.reproject import reproject_video_with_aspect

    size = _canvas(target_aspect=target_aspect, base_resolution=base_resolution)
    vid_path = await resolve_input_async(video, suffix=".mp4")
    out_path = project.output_path("reproject_video", ".mp4", kind=project.VIDEO)
    info = await asyncio.to_thread(
        reproject_video_with_aspect,
        video_path=vid_path, island_metadata={"x": x, "y": y, "width": width, "height": height},
        output_path=out_path,
        target_aspect=f"{size[0]}x{size[1]}" if size else target_aspect or "16:9",
        base_resolution=max(size) if size else base_resolution or 1920,
        crf=crf, codec=codec,
    )
    return {
        "output_video": out_path,
        "info": {k: info.get(k) for k in ("frames_processed", "fps", "canvas_size", "island_position")},
    }


# ---------------------------------------------------------------------------
# TextureFlow — our own Modal app `wzrd-textureflow` (env `wzrd`, source in
# deploy/textureflow/). Local files travel as bytes in `files`, URLs pass
# through, the returned mp4 bytes land in the project's generated/ folder.
# Only caller-set args are sent: the server owns the defaults (api.yaml) and
# all validation. Contract: docs/reference/textureflow.md.
# ---------------------------------------------------------------------------
_TF_MEDIA = ("images", "control_input", "diffusion_mask")
_TF_LOCAL_ARGS = ("ctx", "region", "allow_over_cap", "hap", "recipe")  # never sent to the server
_TF_USD_PER_S = 1.95 / 3600  # Modal L40S list price
_TF_EST_FULL_USD = 0.20  # pre-call guess, defaults (~6 min warm)
_TF_EST_QUICK_USD = 0.03  # pre-call guess, no upscale passes


def _tf_pack_media(args: dict) -> dict[str, bytes]:
    """Swap local paths in the media args for keys into a `files` dict.

    URLs are left as-is (the server downloads them). Keys are basenames, made
    unique, so the server sees the original file names; photos go upright
    (HEIC decoded, EXIF rotation applied — the server would ignore both).
    """
    from wzrd.utils import upright_copy

    files: dict[str, bytes] = {}

    def pack(ref: str) -> str:
        if ref.startswith(("http://", "https://")):
            return ref
        path = Path(ref).expanduser()
        if not path.is_file():
            raise ToolError(f"texture_flow: {ref!r} is neither an http(s) URL nor an existing local file")
        src = Path(upright_copy(path))
        key, n = f"{path.stem}{src.suffix}", 2
        while key in files:
            key, n = f"{path.stem}_{n}{src.suffix}", n + 1
        files[key] = src.read_bytes()
        return key

    for name in _TF_MEDIA:
        if name in args:
            v = args[name]
            args[name] = [pack(x) for x in v] if isinstance(v, list) else pack(v)
    return files


def _tf_target() -> tuple[str, str]:
    return os.getenv("TEXTUREFLOW_APP", "wzrd-textureflow"), os.getenv("WZRD_MODAL_ENV", "wzrd")


@tool(title="TextureFlow animation", tags={"generate", "textureflow"}, open_world=True,
      project_arg=True, render=True)
async def texture_flow(
    images: Optional[list[str]] = None,
    n_seconds: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    base_model: Optional[str] = None,
    use_controlnet1: Optional[bool] = None,
    control_input: Optional[str] = None,
    diffusion_mask: Optional[str] = None,
    preprocessor1: Optional[str] = None,
    controlnet_strength1: Optional[float] = None,
    denoise: Optional[float] = None,
    control_input_fit_strategy: Optional[str] = None,
    map_shape_input_to_ip_masks: Optional[bool] = None,
    mapping_mode: Optional[str] = None,
    n_steps: Optional[int] = None,
    motion_scale: Optional[float] = None,
    use_upscale: Optional[bool] = None,
    upscale_resolution: Optional[int] = None,
    upscale_esrgan: Optional[bool] = None,
    seed: Optional[int] = None,
    output_codec: Optional[str] = None,
    region: Optional[str] = None,
    allow_over_cap: bool = False,
    hap: bool = True,
    recipe: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Render an abstract, morphing, seamlessly flowing animation from 1-6 style images (TextureFlow: SD1.5 + AnimateDiff + IP-Adapter + ControlNet, on WZRD's own Modal GPU).

    For textures, VJ loops, animated logos and per-region fills. It cannot tell
    a story and takes no keyframes: style images drive textures and colors but
    never appear verbatim. Load skill `textureflow` for settings. A saved
    look from the library (list_recipes) renders with recipe="<name>".

    Omit an argument to get the server default. Defaults render full-HD
    upscaled (~6 min warm); for tests set use_upscale=false AND
    upscale_esrgan=false (~30 s at 384x384). A cold GPU adds ~1 min
    (textureflow_warm before a show). Media args take a local path or URL.
    The mp4 lands in the media library (dest=); the preview videos
    (first_pass_video, control_signal_preview, mapping_motion_preview) are
    scratch under the library's .wzrd/intermediates/.

    Args:
        images: 1-6 style images driving content, textures and colors. Required unless a recipe supplies them.
        n_seconds: Length in seconds (2-24, default 6). ~3 s per style image.
        width: Pixels (320-1280, step 32, default 640). High values are slow and can duplicate content.
        height: Pixels (320-1280, step 32, default 640).
        base_model: "SD15/juggernaut_reborn.safetensors" (realistic, default), "SD15/darkSushiMixMix_225D.safetensors" (creative, soft pastel), "SD15/protogenV22Anime_protogenV22.safetensors" (anime, flat colors).
        use_controlnet1: Enable shape guidance from control_input (default false).
        control_input: Image/GIF/video giving shape (logo, surface photo, motion clip, QR). Also the img2img start when denoise < 1, and what shows through black diffusion_mask areas — even with use_controlnet1=false.
        diffusion_mask: Image/video mask: white = animated, black = keeps the shape input (stays black only when there is no control_input). Edge is grown and blurred. Use for region/island videos.
        preprocessor1: With use_controlnet1: "Scribble_XDoG_Preprocessor" (default, rough shape), "CannyEdgePreprocessor" (strong edges), "DepthAnythingV2Preprocessor" (depth), "AnyLineArtPreprocessor_aux" (lineart), "OpenposePreprocessor" (needs a visible person, else no guidance), "none" (luminance/QR mode).
        controlnet_strength1: 0.0-1.0, default 0.45. Subtle 0.35-0.45, strong 0.45-0.65; higher tends to look bad.
        denoise: 0.1-1.0, default 1.0 (fresh render; shape only via controlnet). Below 1 = img2img from control_input, so its colors and layout carry over (0.7-0.9 "AI-skins" a photo or surface). NEVER < 1 without a control_input: the start is black, so the video comes out near-black. Coarse at low n_steps (at 4 steps, 0.81-0.99 all render the same).
        control_input_fit_strategy: "fill / crop" (default), "stretch", "pad".
        map_shape_input_to_ip_masks: Map style images onto the flat color clusters of the shape input (only for flat single-color regions). Overrides mapping_mode.
        mapping_mode: Motion pattern: concentric_circles_inwards/outwards (default outwards), concentric_rectangles_inwards/outwards, rotating_segments_clockwise/counter_clockwise (avoid unless asked), pushing_segments_clockwise/counter_clockwise, vertical_stripes_left/right, horizontal_stripes_up/down.
        n_steps: LCM steps (4-14, default 8). Runtime scales linearly.
        motion_scale: Motion strength (0.7-1.4, default 1.1). 0.9 subtle, 1.25 lively; >1.3 rarely good.
        use_upscale: HD latent-upscale second pass (default true).
        upscale_resolution: Long side for the latent upscale (1024-1536, step 64, default 1280).
        upscale_esrgan: ESRGAN to 1920 px long side (default true; sharp/realistic, less ideal for organic looks). Runs even with use_upscale=false.
        seed: 0-2147483647. Omit for random; the used seed is returned, pass it back to reproduce.
        output_codec: "h264" (default) or "hevc10" (smaller, higher fidelity, not decodable everywhere).
        region: Optional pack layer id this render is for (e.g. "fabric_left"). Recorded in the render's record with that layer's box (resolved against this call's project).
        allow_over_cap: Only after Xander OKs it: run even though it takes the project past its spend cap.
        hap: Also write the engine-playable HAP clip beside the mp4 (`clip` in the result; same render id) — default true.
        recipe: A style recipe name (list_recipes): its style images and settings become the defaults; every argument you pass wins. Add width/height (the region's aspect), and control_input or diffusion_mask for the target.
    """
    # First statement on purpose: locals() is exactly the tool's parameters here.
    args = {k: v for k, v in locals().items() if k not in _TF_LOCAL_ARGS and v is not None}
    import modal

    if recipe:
        from .recipe_tools import apply_recipe

        args = apply_recipe(recipe, args)
    elif not images:
        raise ToolError("texture_flow needs images= (1-6 style images) or recipe=")
    sent = dict(args)  # local paths, before packing — what the record keeps
    region_info = provenance.region_meta(region) if region else {}
    quick = args.get("use_upscale") is False and args.get("upscale_esrgan") is False
    provenance.check_spend(_TF_EST_QUICK_USD if quick else _TF_EST_FULL_USD, allow_over_cap)

    files = await asyncio.to_thread(_tf_pack_media, args)
    app_name, env = _tf_target()
    mb = sum(len(b) for b in files.values()) / 1e6
    log_progress("texture_flow", f"Rendering on {env}/{app_name} ({len(files)} local file(s), {mb:.1f} MB)...")
    tf = modal.Cls.from_name(app_name, "TextureFlow", environment_name=env)()
    res = await with_keepalive(tf.render.remote.aio(args=args, files=files), ctx, "TextureFlow")

    # The main video is the library item; the previews are scratch beside the job.
    response: dict = {}
    scratch = library.intermediates_dir() / (jobs.current_job.get() or f"texture_flow_{project.stamp()}")
    for key, out in res["outputs"].items():
        ext = Path(out["filename"]).suffix or ".mp4"
        if key == "output_video":
            path = library.output_path("texture_flow", ext)
        else:
            scratch.mkdir(parents=True, exist_ok=True)
            path = str(scratch / f"{key}{ext}")
        await asyncio.to_thread(Path(path).write_bytes, out["bytes"])
        response[key] = path
    response.update(seed=res["params"].get("seed"), params=res["params"],
                    request_id=res["request_id"], timings=res["timings"])
    # Cost = billed container seconds on the L40S, all on the main output.
    t = res["timings"]
    seconds = (t.get("total_s") or 0) + ((t.get("restore_s") or 0) if t.get("cold_start") else 0)
    main = response.get("output_video")
    response["_provenance"] = {
        "paid": True,
        "est_usd": {str(Path(main).resolve()): seconds * _TF_USD_PER_S} if main else {},
        "args": {**(sent if recipe else {}), "seed": res["params"].get("seed"),
                 "model": f"{app_name} (TextureFlow)"},
        **region_info,
    }
    return await attach_clip(response, "output_video", hap)


@tool(title="Keep TextureFlow GPU warm", tags={"generate", "textureflow"}, idempotent=True, open_world=True)
async def textureflow_warm(minutes: float = 60, ctx: Optional[Context] = None) -> dict:
    """Keep one TextureFlow GPU container running so renders skip the cold start (normally +15-20 s; the first after a redeploy take 2-3 min).

    Calling again moves the deadline; minutes=0 cools down now. The timer runs
    on Modal, so it scales back to zero even if this machine goes offline.
    An idle warm L40S costs ~$2/hour — cool down when done.

    Args:
        minutes: How long to stay warm from now (0-720). 0 = scale to zero now.
    """
    import modal

    app_name, env = _tf_target()
    fn = modal.Function.from_name(app_name, "set_warm", environment_name=env)
    return await fn.remote.aio(minutes)


# ---------------------------------------------------------------------------
# Layer pack — the offline ↔ realtime handoff
# ---------------------------------------------------------------------------
_LAYERPACK_TAGS_SCHEMA = {
    "type": "object",
    "properties": {
        "layers": {
            "type": "object",
            "description": "Per-mask metadata keyed by mask id (or filename).",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "parent": {"type": "string"},
                    "z": {"type": "integer"},
                },
            },
        },
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "members": {"type": "array", "items": {"type": "string"}}},
                "required": ["id", "members"],
            },
        },
    },
}


@tool(title="Build layer pack", tags={"regions", "engine"}, project_arg=True)
async def build_layerpack(
    masks: list[str],
    surface_image: Optional[str] = None,
    mask_ids: Optional[list[str]] = None,
    tags: Annotated[
        Optional[dict[str, Any]],
        Field(description="Optional per-layer labels/tags/parent/z and layer groups.",
              json_schema_extra=_LAYERPACK_TAGS_SCHEMA),
    ] = None,
    projector_width: int = 0,
    projector_height: int = 0,
    source_capture: Optional[str] = None,
    canny: Optional[str] = None,
    feather_px: int = 1,
    include_background: bool = False,
    pack_name: Optional[str] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Build a layer pack (pack.json + masks/ + references/) in the project's packs/ — what the realtime engine loads.

    Each mask becomes one addressable layer. Give stable `mask_ids` so engine
    bindings keep targeting the same layer across re-shoots/re-segmentations.
    Masks straight from extract_color_regions carry their region's padded
    `source_box` into the pack (where the engine places that region's clips).
    If the project has no scene.json yet, a starter one pointing at this pack
    is written (an existing one is never touched). In a project with a canvas the pack is authored at the canvas (the engine
    scales it to the projector): leave projector_width/height unset, and the
    surface_image must be canvas-sized.

    Args:
        masks: Mask images (paths/URLs), typically extract_color_regions' region_mask files. Max 256.
        surface_image: The darkened/aligned surface (preview overlay; sets projector resolution if not given).
        mask_ids: Semantic ids parallel to `masks` (same length); default = mask file stems.
        projector_width: Projects without a canvas only: pixels; defaults to the surface image width.
        projector_height: Projects without a canvas only: pixels; defaults to the surface image height.
        source_capture: Original photo, copied into references/.
        canny: Edge aid image, copied into references/.
        feather_px: Mask edge feathering radius (0 = hard edges).
        include_background: Also generate `000_background.png`, the complement of all masks.
        pack_name: Folder name under packs/; default timestamped.
    """
    from wzrd.layerpack import build_layerpack as _build, islands_source_boxes

    if mask_ids and len(mask_ids) != len(masks):
        raise ToolError("mask_ids length must match masks length")
    size = _canvas(projector_width=projector_width, projector_height=projector_height)

    resolved_masks = await asyncio.gather(*(resolve_input_async(m, suffix=".png") for m in masks))

    async def _opt(ref: Optional[str]) -> Optional[str]:
        return await resolve_input_async(ref, suffix=".png") if ref else None

    surf_path, src_path, cny_path = await asyncio.gather(_opt(surface_image), _opt(source_capture), _opt(canny))
    if size and surf_path:
        _check_canvas_image(surf_path, size, "surface_image")

    # Stage masks with stable, ordered filenames so layerpack discovery sees
    # them in the caller's order.
    masks_dir = Path(mkdtemp(prefix="wzrd_masks_")) / "masks_in"
    masks_dir.mkdir()
    ordered: list[str] = []
    for i, src in enumerate(resolved_masks):
        dest = f"{i:03d}_{mask_ids[i] if mask_ids else f'layer_{i:03d}'}.png"
        copyfile(src, masks_dir / dest)
        ordered.append(dest)

    # Rekey caller per-mask metadata to the staging filenames so it applies.
    rekeyed = dict(tags or {})
    if mask_ids:
        src_layers = rekeyed.get("layers") or {}
        rekeyed["layers"] = {
            ordered[i]: {**src_layers.get(stem, {}), "id": src_layers.get(stem, {}).get("id", stem)}
            for i, stem in enumerate(mask_ids)
        }

    # C4: each mask's padded region box, from the islands.json beside it
    # (staging drops the folder, so look it up here). Caller tags win.
    layers = dict(rekeyed.get("layers") or {})
    boxes: dict[Path, dict[str, list[int]]] = {}
    for i, src in enumerate(resolved_masks):
        folder = Path(src).parent
        if folder not in boxes:
            boxes[folder] = islands_source_boxes(folder)
        box = boxes[folder].get(Path(src).name)
        if box and "source_box" not in layers.get(ordered[i], {}):
            layers[ordered[i]] = {**layers.get(ordered[i], {}), "source_box": box}
    if layers:
        rekeyed["layers"] = layers

    projector_resolution = size or (
        (projector_width, projector_height) if projector_width > 0 and projector_height > 0 else None
    )

    # The pack lands in the project so render-core loads it straight off disk.
    pack_dir = Path(project.output_dir("build_layerpack", kind=project.PACKS, name=pack_name or ""))
    log_progress("build_layerpack", "Building layer pack...")
    scene = await asyncio.to_thread(
        _build,
        masks_dir=str(masks_dir), output_dir=str(pack_dir), surface=surf_path, tags=rekeyed,
        projector_resolution=projector_resolution, source_capture=src_path, canny=cny_path,
        feather_px=feather_px, include_background=include_background, mask_frame=size,
    )
    scene_json, created = _starter_scene(pack_dir)
    result = {
        "scene": scene,
        "pack_dir": str(pack_dir),
        "pack_json": str(pack_dir / "pack.json"),
        "layer_count": len(scene["layers"]),
        "projector_resolution": scene["projector_resolution"],
        "scene_json": scene_json,
        "scene_json_created": created,
    }
    if not created:
        result["note"] = "scene.json already existed and was left untouched — it may point at another pack"
    return result


def _starter_scene(pack_dir: Path) -> tuple[str, bool]:
    """Write `<project>/scene.json` pointing at *pack_dir* unless one exists.

    The engine refuses to boot without a scene + pack and nothing else writes
    a scene offline, so without this a fresh project can't be opened.
    """
    import json

    path = project.project_dir() / "scene.json"
    starter = {
        "version": 1,
        "pack": pack_dir.relative_to(project.project_dir()).as_posix(),
        "transport": {"bpm": 120},
        "bindings": [],
        "post": [],
    }
    try:
        with open(path, "x") as f:  # "x": never overwrite, even in a race
            json.dump(starter, f, indent=2)
            f.write("\n")
    except FileExistsError:
        return str(path), False
    return str(path), True


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
@tool(title="Simulate projected view", tags={"preview"}, project_arg=True)
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
    """Predict what a viewer will see: composite projector output (additive light on black) onto the surface photo, modulated by the surface's reflectivity.

    Args:
        projector_video: Projector output (a background-subtracted video, or raw TextureFlow output).
        surface_image: The surface photo, ideally with non-reflective areas (sky…) masked to pure black.
        surface_weight: How visible the bare surface texture is (0.0-1.0). Lower = more projection contrast.
        projection_strength: Projected light multiplier (0.5-2.0); >1 simulates a brighter projector.
        ambient: Reflectivity floor for dark-but-not-black surface pixels (0.0-0.3).
        codec: ffmpeg video codec.
        crf: Quality — lower is better (0-51).
    """
    from wzrd.simulate_view import simulate_view as _simulate_view

    vid_path, surf_path = await asyncio.gather(
        resolve_input_async(projector_video, suffix=".mp4"),
        resolve_input_async(surface_image, suffix=".png"),
    )
    out_path = project.output_path("simulate_view", ".mp4", kind=project.VIDEO)
    info = await asyncio.to_thread(
        _simulate_view,
        projector_video_path=vid_path, surface_image_path=surf_path, output_path=out_path,
        surface_weight=surface_weight, projection_strength=projection_strength,
        ambient=ambient, codec=codec, crf=crf,
    )
    return {
        "simulated_view_video": out_path,
        "info": {k: info.get(k) for k in ("frames_processed", "fps", "video_size", "surface_weight", "projection_strength")},
    }
