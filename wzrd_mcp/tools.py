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

from . import jobs, library, project, provenance, textureflow_local, tf_queue, tf_spec
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


def _surface_run(folder: Path) -> str:
    """A fresh `YYYYmmdd-HHMMSS[-n]` run id with none of its three files taken."""
    base, n = project.stamp(), 1
    run = base
    while any((folder / f"{run}_{k}.png").exists() for k in ("lit", "surface", "ambient")):
        n += 1
        run = f"{base}-{n}"
    return run


_CORNERS_DOC = ("Manual quad in the lit photo: 4 normalised [x, y] (0-1), TL, TR, BR, BL — used exactly "
                "as given (no margin). Omit = auto-detect. A result's `corners` round-trip here.")


@tool(title="Rectify surface", tags={"surface"}, project_arg=True)
async def rectify_surface(
    lit_photo: str,
    ambient_photo: Optional[str] = None,
    corners: Annotated[
        Optional[list[Annotated[list[float], Field(min_length=2, max_length=2)]]],
        Field(min_length=4, max_length=4, description=_CORNERS_DOC),
    ] = None,
    margin: float = 0.01,
    min_projection_fraction: float = 0.10,
    max_brightness: float = 0.20,
    detail_boost: float = 1.15,
    chroma_correction: float = 0.5,
    gradient_source: float = 0.5,
    guide_fraction: float = 0.04,
    ctx: Optional[Context] = None,
) -> dict:
    """Venue photos → the show's three surface images, one run in surfaces/: <run>_lit.png (rectified lit crop), <run>_surface.png (dark colour-corrected surface), <run>_ambient.png (aligned ambient photo).

    The projector's lit quad in the lit photo (full white projected) is
    rectified to exactly the project canvas (legacy project: 1920x1080).
    With an ambient photo (same view, room lit) it is aligned to the lit crop
    and darkened; without one the lit crop is darkened and surface_ambient
    is null. Pin the keepers with pin_asset (surface_lit, surface,
    surface_ambient); re-run with nudged `corners` when detection is off.

    Args:
        lit_photo: Photo with the projector showing full white (whole frame — import_photos(conform=false)).
        ambient_photo: Optional photo of the same view with the room lit.
        margin: Auto-detect only: shrink the detected quad inward by this fraction (0-0.1).
        min_projection_fraction: Auto-detect only: the lit area's minimum share of the photo.
        max_brightness: Darkening luminance ceiling (0.0-1.0). Lower = darker surface.
        detail_boost: Darkening detail amplification (1 = keep).
        chroma_correction: How much colour drops with luminance (0 = keep colour, 1 = proportional).
        gradient_source: Tone-curve weighting: 0 = raw-luminance edges, 1 = fine texture.
        guide_fraction: Base/detail split radius as a fraction of the short side.
    """
    from wzrd.prepare_surface import rectify_surface as _rectify

    size = project.canvas() or (1920, 1080)
    coros = [resolve_input_async(lit_photo, suffix=".png")]
    if ambient_photo:
        coros.append(resolve_input_async(ambient_photo, suffix=".png"))
    resolved = await asyncio.gather(*coros)
    log_progress("rectify_surface", "rectify → align → darken" if ambient_photo else "rectify → darken")
    try:
        res = await asyncio.to_thread(
            _rectify, resolved[0], resolved[1] if ambient_photo else None,
            output_size=size, corners=corners, margin=margin,
            min_projection_fraction=min_projection_fraction, max_brightness=max_brightness,
            detail_boost=detail_boost, guide_fraction=guide_fraction,
            chroma_correction=chroma_correction, gradient_source=gradient_source,
        )
    except ValueError as exc:  # no lit quad found, bad corners
        raise ToolError(f"{exc} — pass corners= (TL, TR, BR, BL, normalised) to set the quad by hand")

    folder = project.kind_dir(project.SURFACES)
    run = _surface_run(folder)
    paths = {k: folder / f"{run}_{k}.png" for k in ("lit", "surface", "ambient")}

    def save() -> None:
        res["lit"].save(paths["lit"])
        res["surface"].save(paths["surface"])
        if res["ambient"] is not None:
            res["ambient"].save(paths["ambient"])

    await asyncio.to_thread(save)
    lit_src = str(Path(lit_photo).expanduser().resolve()) if not lit_photo.startswith("http") else None
    amb_src = (str(Path(ambient_photo).expanduser().resolve())
               if ambient_photo and not ambient_photo.startswith("http") else None)
    ambient_out = str(paths["ambient"]) if res["ambient"] is not None else None
    parents = {str(paths["lit"]): lit_src, str(paths["surface"]): amb_src or lit_src}
    if ambient_out:
        parents[ambient_out] = amb_src
    return {
        "run": run,
        "surface_lit": str(paths["lit"]),
        "surface": str(paths["surface"]),
        "surface_ambient": ambient_out,
        "corners": res["corners"],
        "manual_corners": res["manual_corners"],
        "align": res["align"],
        "size": list(res["size"]),
        # The quad actually rectified goes into every output's record (top level
        # and args), so a trial reloaded from an auto run still has its corners.
        "_provenance": {
            "parent": {k: v for k, v in parents.items() if v},
            "args": {"corners": res["corners"], "manual_corners": res["manual_corners"]},
            "record": {"corners": res["corners"], "manual_corners": res["manual_corners"],
                       "run": run, "align": res["align"]},
        },
    }


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
    region), `source_box`, `aspect_ratio`, `color` (the map's hex) and
    `area_fraction` (share of the canvas). A map at another size than the surface (or the project
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
        entry: dict[str, Any] = {"source_box": r.get("source_box"), "color": r.get("color"),
                                 "area_fraction": r.get("area_fraction")}
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
# TextureFlow — our own Modal app `textureflow` (env `wzrd`), deployed from the
# textureflow-lab repo (`deploy.py`; CI redeploys it on every push to its main),
# so renders always run the lab's latest pipeline. `TEXTUREFLOW_APP=wzrd-textureflow`
# falls back to the frozen ComfyUI-graph app (textureflow-lab `comfyui_app/`, kept
# deployed as the rollback). Local files travel as bytes in `files`, URLs pass
# through, the returned mp4 bytes land in the project's generated/ folder.
# The tool is generated from the endpoint's published arg spec (`tf_spec`:
# the deployed app's `spec()`, cached, else the vendored snapshot), merged
# with WZRD's own args below. Only caller-set args are sent: the server owns
# the defaults and all validation. `backend="local"` renders the same call on Xander's desktop
# GPU instead (`textureflow_local.py`, $0); the caller always picks — no
# auto-routing, no fallback. Draft → pick → finish (spec 2.2.0): `draft=true`
# renders only the final's first pass; `finish_from=<request_id>` finishes one
# from the bundle the server kept (24 h, on the backend that rendered it), so
# every record keeps `record.textureflow` (request_id, backend, stage, draft).
# Contract: docs/reference/textureflow.md.
# ---------------------------------------------------------------------------
_TF_MEDIA = ("images", "control_input", "diffusion_mask")
_TF_BACKENDS = ("modal", "local")
_TF_USD_PER_S = 3.03 / 3600  # Modal RTX PRO 6000 list price (textureflow-lab deploy.py GPU)
_TF_EST_FULL_USD = 0.20  # pre-call guess, defaults (~3–4 min warm)
_TF_EST_QUICK_USD = 0.03  # pre-call guess, no upscale passes
_TF_EST_DRAFT_USD = 0.10  # pre-call guess, draft=true: the final's first pass ≈ half a full render
_TF_EST_FINISH_USD = 0.10  # pre-call guess, finish_from: the full render minus its first pass


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


def _tf_backend(backend: Optional[str] = None) -> str:
    """The backend a call renders on: its `backend=`, else $TEXTUREFLOW_BACKEND, else Modal."""
    name = (backend or os.getenv("TEXTUREFLOW_BACKEND") or "modal").strip().lower()
    if name not in _TF_BACKENDS:
        raise ToolError(f"texture_flow: backend={name!r} — use 'modal' (default) or 'local' (the desktop GPU)")
    return name


def _tf_source(request_id: str) -> Optional[dict]:
    """The library's render whose server `request_id` is *request_id* (a draft to finish):
    `{id, path, record}` (`path` = its mp4 if still on disk, else None), or None when no
    record here has it (rendered from another machine, or before records kept the id)."""
    row = library.query_one(
        "SELECT id, record FROM renders WHERE kind='wzrd' AND tool='texture_flow' "
        "AND json_extract(record, '$.textureflow.request_id')=? ORDER BY created DESC LIMIT 1", (request_id,))
    if not row:
        return None
    f = library.query_one("SELECT path FROM files WHERE render_id=? AND variant='master' AND missing=0 "
                          "ORDER BY mtime DESC LIMIT 1", (row["id"],))
    return {"id": row["id"], "path": f["path"] if f else None, "record": library._loads(row["record"], {})}


def _tf_finish_backend(request_id: str, source: Optional[dict], backend: Optional[str], strict: bool = True) -> str:
    """A finish runs where its draft's bundle lives: the source record's backend. An explicit
    `backend=` that contradicts it is refused (the other backend has no such bundle)."""
    recorded = ((source or {}).get("record", {}).get("textureflow") or {}).get("backend")
    if strict and recorded and backend and _tf_backend(backend) != recorded:
        raise ToolError(f"texture_flow: draft {request_id} was rendered on backend={recorded!r} and its "
                        f"bundle only lives there — drop backend= or pass {recorded!r}")
    return _tf_backend(recorded or backend)


def _tf_estimate(args: dict) -> float:
    finish = args.get("finish_from")
    backend = (_tf_finish_backend(finish, _tf_source(finish), args.get("backend"), strict=False) if finish
               else _tf_backend(args.get("backend")))
    if backend == "local":
        return 0.0  # our own GPU: free, never held against the spend cap
    if finish:
        return _TF_EST_FINISH_USD
    if args.get("draft"):
        return _TF_EST_DRAFT_USD
    quick = args.get("use_upscale") is False and args.get("upscale_esrgan") is False
    return _TF_EST_QUICK_USD if quick else _TF_EST_FULL_USD


jobs.ESTIMATORS["texture_flow"] = _tf_estimate


_tf_target = tf_spec.target


# WZRD-owned part of the tool description (between the spec's summary and the Args).
_TF_INTRO = """\
Runs on WZRD's own Modal GPU (default) or, with backend="local", on Xander's
desktop GPU (slower, $0). Load skill `textureflow` for settings. A saved look
from the library (list_recipes) renders with recipe="<name>".

Omit an argument to get the server default. A cold Modal GPU adds ~1 min
(textureflow_warm before a show). Media args take a local path or URL.
The mp4 lands in the media library (dest=); the preview videos
(first_pass_video, control_signal_preview, mapping_motion_preview) are
scratch under the library's .wzrd/intermediates/."""

# Docs of WZRD's own args (hand-written params of `_texture_flow`), and notes
# appended to spec args where WZRD changes their meaning.
_TF_OWN_DOCS = {
    "region": 'Optional pack layer id this render is for (e.g. "fabric_left"). Recorded in the render\'s record with that layer\'s box (resolved against this call\'s project).',
    "allow_over_cap": "Only after Xander OKs it: run even though it takes the project past its spend cap.",
    "hap": "Also write the engine-playable HAP clip beside the mp4 (`clip` in the result; same render id) — default true.",
    "recipe": "A style recipe name (list_recipes): its style images and settings become the defaults; every argument you pass wins. Add output_width/output_height (the region's aspect; density for the content scale), and control_input or diffusion_mask for the target.",
    "backend": '"modal" (WZRD\'s Modal GPU, paid, counts toward the spend cap) or "local" (Xander\'s desktop RTX 3090: $0, same args, outputs and seed handling, renders anything Modal can; the desktop must be on — if it\'s off the call fails in ~2 s). Omitted = $TEXTUREFLOW_BACKEND, else "modal". Xander picks it; never switch backends on your own or retry a failed call on the other one.',
}
_TF_NOTES = {"images": "Required unless a recipe supplies them."}
# Added to the description when the endpoint's spec has `finish_from` (spec ≥ 2.2.0).
_TF_FINISH_INTRO = """\
Draft → pick → finish: render candidates with draft=true at the final's
settings, then finish_from="<request_id>" (from the draft's reply or its
render record) with nothing else but finish settings: no images or recipe;
it runs on the backend that rendered the draft, recorded as its child."""
_TF_FINISH_NOTES = {"images": "Required unless a recipe supplies them or finish_from is set."}


class _TfCall:
    """One texture_flow call's resolved inputs (shared by a fresh call and a re-adopted local job)."""

    def __init__(self, args: dict, sent: dict, backend: str, finish: Optional[str], source: Optional[dict],
                 recipe: Optional[str], region_info: dict, warnings: list[str], hap: Any):
        self.args, self.sent, self.backend, self.finish = args, sent, backend, finish
        self.source, self.recipe, self.region_info, self.warnings, self.hap = source, recipe, region_info, warnings, hap
        self.moved = False  # the operator moved it to the other backend while it was queued

    @property
    def eta_args(self) -> dict:
        """What the ETA model reads: a finish renders at its draft's resolved params."""
        base = ((self.source or {}).get("record", {}).get("textureflow") or {}).get("params") or {}
        return {**base, **self.sent, **({"finish_from": self.finish, "draft": False} if self.finish else {})}

    @property
    def label(self) -> str:
        if self.finish:
            return f"finish {self.finish}"
        if self.recipe:
            return f"recipe {self.recipe}"
        if self.sent.get("prompt"):
            return str(self.sent["prompt"])[:80]
        imgs = self.sent.get("images") or []
        return Path(str(imgs[0])).name if imgs else "texture_flow"


def _tf_prepare(region: Optional[str] = None, hap: Any = True, recipe: Optional[str] = None,
                backend: Optional[str] = None, **spec_args: Any) -> _TfCall:
    """Validate + resolve a call's args (recipe, finish routing, region) — no I/O beyond the library."""
    args = {k: v for k, v in spec_args.items() if v is not None}
    for k in ("allow_over_cap", "ctx"):
        args.pop(k, None)
    images = args.get("images")
    finish = args.get("finish_from") or None
    if not finish:
        args.pop("finish_from", None)  # "" = unset
    warnings: list[str] = []
    source = None  # the draft being finished, when this library has its record
    if finish:
        given = [k for k in _TF_MEDIA if k in args] + (["recipe"] if recipe else [])
        if given:
            raise ToolError(f"texture_flow: finish_from takes no {', '.join(given)} — the draft's own inputs "
                            "and settings are reused; only finish settings (upscale, output size, codec, …) may be set")
        source = _tf_source(finish)
        backend = _tf_finish_backend(finish, source, backend)
    else:
        backend = _tf_backend(backend)
        if recipe:
            from .recipe_tools import apply_recipe

            args = apply_recipe(recipe, args, warnings)
        elif not images:
            raise ToolError("texture_flow needs images= (style images) or recipe=")
    region_info = provenance.region_meta(region) if region else {}
    return _TfCall(args, dict(args), backend, finish, source, recipe, region_info, warnings, hap)


_TF_PROGRESS_DICT = "textureflow-progress"  # Modal Dict the endpoint writes {stage, step, total, t} into (Q5)
# Its lifecycle entries around the render's own stages: started (+cold_start) → stages → saving → done | failed.
_TF_DICT_LIFECYCLE = ("started", "saving", "done", "failed")


async def _tf_modal(args: dict, files: dict, job: Optional[str], ctx: Optional[Context]) -> dict:
    """Spawn on the Modal app and wait; live progress from the Dict when the deployed spec() says so."""
    import modal

    app_name, env = _tf_target()
    mb = sum(len(b) for b in files.values()) / 1e6
    log_progress("texture_flow", f"Rendering on {env}/{app_name} ({len(files)} local file(s), {mb:.1f} MB)...")
    tf = modal.Cls.from_name(app_name, "TextureFlow", environment_name=env)()
    key = job if (job and "progress_dict" in tf_spec.capabilities()) else None
    call = await tf.render.spawn.aio(args=args, files=files, **({"progress_key": key} if key else {}))
    jobs.set_remote(f"modal|{call.object_id}", call.cancel.aio)
    tf_queue.report(job, phase="starting" if key else "rendering")
    task = asyncio.ensure_future(call.get.aio())
    if not key:
        return await with_keepalive(task, ctx, "TextureFlow")

    async def follow() -> dict:
        d = modal.Dict.from_name(_TF_PROGRESS_DICT, environment_name=env)
        failed = False
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=1.0)
                if done:
                    return task.result()
                try:
                    v = await d.get.aio(key)
                except Exception as exc:  # noqa: BLE001 — progress is best-effort
                    if not failed:
                        log_progress("texture_flow", f"progress Dict unreadable ({type(exc).__name__}) — estimate only")
                    failed, v = True, None
                name = v.get("stage") if isinstance(v, dict) else None
                if name == "started":  # the container picked the call up (lifecycle entries aren't stages)
                    tf_queue.report(job, phase="starting")
                elif name and name not in _TF_DICT_LIFECYCLE:
                    tf_queue.report(job, stage=name, step=v.get("step"), total=v.get("total"))
        finally:
            if not task.done():
                task.cancel()
            try:
                await d.pop.aio(key)
            except Exception:  # noqa: BLE001
                pass

    return await with_keepalive(follow(), ctx, "TextureFlow")


async def _tf_collect(res: dict, call: _TfCall, backend: str, job: Optional[str]) -> dict:
    """The server's render dict → files in the library + the reply + `_provenance` (both backends)."""
    tf_queue.report(job, phase="saving")
    # The main video is the library item; the previews are scratch beside the job.
    # The file is named after the server's request_id (it carries the timestamp), so a
    # file on disk alone leads back to its draft bundle.
    response: dict = {}
    request_id = res.get("request_id")
    scratch = library.intermediates_dir() / (job or f"texture_flow_{project.stamp()}")
    for key, out in res["outputs"].items():
        ext = Path(out["filename"]).suffix or ".mp4"
        if key == "output_video":
            path = library.output_path("texture_flow", ext, name=f"texture_flow_{request_id}" if request_id else "")
        else:
            scratch.mkdir(parents=True, exist_ok=True)
            path = str(scratch / f"{key}{ext}")
        await asyncio.to_thread(Path(path).write_bytes, out["bytes"])
        response[key] = path
    info = res.get("info") or {}
    finish, source = call.finish, call.source
    response.update(seed=res["params"].get("seed"), params=res["params"],
                    request_id=request_id, timings=res["timings"])
    # Spec ≥ 2.2.0: the bundle a later finish_from=<request_id> finishes (servers before it send none).
    stage = info.get("stage") or ("finish" if finish else "draft" if call.args.get("draft") else "full")
    bundle = {k: info["draft"].get(k) for k in ("id", "expires_at")} if isinstance(info.get("draft"), dict) else None
    if bundle:
        response["draft"] = bundle
    for k in ("finished_from", "finish_warnings"):
        if info.get(k):
            response[k] = info[k]
    # Cost = billed container seconds on the endpoint's GPU, all on the main
    # output; a local render costs nothing.
    t = res["timings"]
    seconds = (t.get("total_s") or 0) + ((t.get("restore_s") or 0) if t.get("cold_start") else 0)
    main = response.get("output_video")
    paid = backend == "modal"
    if paid:
        model = f"{_tf_target()[0]} (TextureFlow)"
    else:
        model = f"textureflow-local (TextureFlow, {info.get('gpu') or tf_queue.S.local.gpu or 'desktop GPU'})"
    rec_args = dict(call.sent) if call.recipe else {}
    if source:  # a finish records the draft's settings too (More like this, the Settings list)
        drop = {"draft", "seed", "model", "finish_from"}
        rec_args = {**{k: v for k, v in (source["record"].get("args") or {}).items() if k not in drop}, **call.sent}
    if call.moved:  # the record says where it really rendered, not the backend it was submitted with
        rec_args["backend"] = backend
    # Q7: the stage timings + sizes are the ETA model's training data (tf_eta) and the lab's dataset.
    sizes = {"stage_s": info.get("stage_s"), "gpu": info.get("gpu"), "first_pass_size": info.get("size"),
             "output_size": info.get("final_size"), "n_frames": info.get("n_frames"),
             "n_output_frames": info.get("n_output_frames"), "steps": info.get("steps"), "timings": t}
    response["_provenance"] = {
        "paid": paid,
        "est_usd": {str(Path(main).resolve()): seconds * _TF_USD_PER_S} if main and paid else {},
        "args": {**rec_args, "seed": res["params"].get("seed"), "model": model},
        "record": {"textureflow": {
            "request_id": request_id, "backend": backend, "stage": stage, "draft": bundle,
            "params": res["params"], **({"finished_from": finish} if finish else {}), **sizes,
        }},
        **call.region_info,
    }
    if source:  # lineage: draft → final (the draft's render id, even if its mp4 is gone)
        response["_provenance"]["parent"] = source["id"]
    if call.warnings:
        response["warnings"] = call.warnings
    return await attach_clip(response, "output_video", call.hap)


async def _texture_flow(
    region: Optional[str] = None,
    allow_over_cap: bool = False,
    hap: bool = True,
    recipe: Optional[str] = None,
    backend: Optional[str] = None,
    ctx: Optional[Context] = None,
    **spec_args: Any,
) -> dict:
    """The body of `texture_flow`; its signature + docstring are generated (`_tf_tool`).

    Checks (args, spend cap) → tracked by `tf_queue` → Modal: wait for a lane slot, spawn; local: hold
    while the desktop is unreachable, submit (the server queues it) → files into the library. A queued
    job the operator moves re-routes here (`tf_queue.Moved` / `acquire_modal` returning "local")."""
    call = _tf_prepare(region=region, hap=hap, recipe=recipe, backend=backend, **spec_args)
    args = call.args
    if call.backend == "local":
        textureflow_local.config()  # unconfigured: a clear error now
        if not tf_queue.ssh_host():
            # Without SSH the hub can't wake the desktop: one that is off fails here (≈ 2 s), before
            # the spend check — even a wait=false caller sees it.
            await textureflow_local.health()
    est = {"modal": _tf_estimate({**args, "backend": "modal"}) if not call.finish else _TF_EST_FINISH_USD,
           "local": 0.0}
    files = await asyncio.to_thread(_tf_pack_media, args)
    job = jobs.current_job.get()
    if job:  # in its lane from here (status "queued"), so a wait=false reply already says so
        await tf_queue.track(job, call.backend, call.eta_args, pinned=bool(call.finish), est=est,
                             allow_over_cap=allow_over_cap, label=call.label)
    try:
        provenance.check_spend(est[call.backend], allow_over_cap)
        backend_now = call.backend
        while True:
            entry = tf_queue.entry(job)
            backend_now = entry.backend if entry else backend_now
            if backend_now == "modal":
                if job and await tf_queue.acquire_modal(job) != "modal":
                    continue  # moved to local while it waited
                try:
                    res = await _tf_modal(args, files, job, ctx)
                finally:
                    tf_queue.release(job or "")
                break
            try:
                if job:
                    await tf_queue.hold_local(job)
                mb = sum(len(b) for b in files.values()) / 1e6
                log_progress("texture_flow", f"Rendering on the local GPU ({len(files)} local file(s), {mb:.1f} MB)...")
                res = await textureflow_local.render(args, files, ctx, job=job,
                                                     meta={"client": "wzrd", "label": call.label})
                tf_queue.release(job or "")
                break
            except tf_queue.Moved:
                tf_queue.consume_move(job or "")
                log_progress("texture_flow", f"job {job} moved to Modal")
        call.moved = backend_now != call.backend
        return await _tf_collect(res, call, backend_now, job)
    finally:
        if job:
            tf_queue.finish(job)


async def adopt_local(row: dict) -> None:
    """Hub start (render-queue Q8): finish a local job a previous hub submitted — follow the server job,
    save its files, record them, end the job row. Runs as a hub task (`tf_queue.adopt_all`)."""
    from . import _log

    job, t0 = row["id"], float(row["started"])
    client_args = library._loads(row["args"], {})
    server_id = (row["remote"] or "").split("|", 1)[1]
    tokens = [(jobs.current_job, jobs.current_job.set(job)),
              (library.current_dest, library.current_dest.set(row.get("dest"))),
              (project.call_project, project.call_project.set(row.get("project")))]
    try:
        try:
            call = _tf_prepare(**{k: v for k, v in client_args.items() if k != "allow_over_cap"})
        except Exception:  # noqa: BLE001 — a recipe gone since: keep what the job row has
            call = _TfCall(dict(client_args), dict(client_args), "local", client_args.get("finish_from"),
                           None, None, {}, [], client_args.get("hap", True))
        call.backend = "local"
        await tf_queue.track(job, "local", call.eta_args, pinned=True, est={"local": 0.0}, label=call.label)
        e = tf_queue.entry(job)
        if e:
            e.server_id, e.local = server_id, "queued"
        await tf_queue.hold_local(job)
        res = await textureflow_local.collect(server_id, None, job)
        tf_queue.release(job)
        result = await _tf_collect(res, call, "local", job)
        result, records, cost = await asyncio.to_thread(_log._with_provenance, "texture_flow", client_args,
                                                        result, t0, job)
        jobs.finish(job, ok=True, outputs=[r["path"] for r in records], est_usd=cost)
        log_progress("texture_flow", f"re-adopted job {job} finished: {result.get('output_video')}")
    except asyncio.CancelledError:
        jobs.finish(job, ok=False, error="cancelled", cancelled=True)
        raise
    except Exception as exc:  # noqa: BLE001
        jobs.finish(job, ok=False, error=str(exc))
        log_progress("texture_flow", f"re-adopted job {job} failed: {exc}")
    finally:
        tf_queue.finish(job)
        jobs.forget(job)
        for var, token in reversed(tokens):
            var.reset(token)


def _tf_tool():
    """`texture_flow` = `_texture_flow` with the spec's args as its parameters and docs (`tf_spec`)."""
    import inspect

    spec = tf_spec.current().spec
    own = list(inspect.signature(_texture_flow, eval_str=True).parameters.values())[:-1]  # minus **spec_args
    skip = {p.name for p in own}
    params = tf_spec.parameters(spec, skip) + own

    async def texture_flow(**kwargs):
        return await _texture_flow(**kwargs)

    texture_flow.__signature__ = inspect.Signature(params, return_annotation=dict)
    texture_flow.__annotations__ = {**{p.name: p.annotation for p in params}, "return": dict}
    finish = "finish_from" in spec["parameters"]
    notes = {**_TF_NOTES, **(_TF_FINISH_NOTES if finish else {})}
    intro = _TF_INTRO + ("\n\n" + _TF_FINISH_INTRO if finish else "")
    lines = tf_spec.args_doc(spec, notes, skip) + [f"{k}: {v}" for k, v in _TF_OWN_DOCS.items()]
    texture_flow.__doc__ = (spec["summary"].strip() + "\n\n" + intro + "\n\nArgs:\n"
                            + "\n".join(f"    {line}" for line in lines))
    return tool(title="TextureFlow animation", tags={"generate", "textureflow"}, open_world=True,
                project_arg=True, render=True)(texture_flow)


texture_flow = _tf_tool()


@tool(title="Keep TextureFlow GPU warm", tags={"generate", "textureflow"}, idempotent=True, open_world=True)
async def textureflow_warm(minutes: float = 60, backend: Optional[str] = None,
                           ctx: Optional[Context] = None) -> dict:
    """Keep one TextureFlow GPU container running so renders skip the cold start (normally +15-20 s; the first after a redeploy take 2-3 min).

    Calling again moves the deadline; minutes=0 cools down now. The timer runs
    on Modal, so it scales back to zero even if this machine goes offline.
    An idle warm GPU (RTX PRO 6000) costs ~$3/hour — cool down when done.
    backend="local" only checks that the desktop GPU is reachable (it is
    always warm while its service runs; minutes is ignored, Modal untouched).

    Args:
        minutes: How long to stay warm from now (0-720). 0 = scale to zero now.
        backend: "modal" (default: $TEXTUREFLOW_BACKEND, else "modal") or "local" (the desktop; must be on).
    """
    if _tf_backend(backend) == "local":
        info = await textureflow_local.health()
        return {"warm": info.get("worker") != "down", "backend": "local",
                **{k: info.get(k) for k in ("gpu", "worker", "busy", "queue")}}
    import modal

    app_name, env = _tf_target()
    fn = modal.Function.from_name(app_name, "set_warm", environment_name=env)
    out = await fn.remote.aio(minutes)
    tf_queue.note_warm(minutes)
    return out


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
