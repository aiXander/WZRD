"""MCP tool wrappers that bridge MCP ↔ wzrd functions.

Each tool handles: resolve inputs (URL→local) → call wzrd → publish outputs.
"""

from __future__ import annotations

from typing import Optional

from fastmcp import Context

from .file_io import make_temp_dir, make_temp_path, resolve_input, upload
from .server import mcp


# ---------------------------------------------------------------------------
# Tool 1: subtract_background_frame
# ---------------------------------------------------------------------------
@mcp.tool()
async def subtract_background_frame(
    generated_image: str,
    background_image: str,
    threshold: int = 10,
    ramp: int = 20,
    gamma: float = 0.85,
    blur_radius: float = 0.004,
    ctx: Optional[Context] = None,
) -> dict:
    """Remove background from a single image, extracting the foreground subject for additive projection mapping.

    Uses luminance-based difference with soft ramp thresholding and Gaussian feathering.

    Args:
        generated_image: URL or path to the generated image with subject.
        background_image: URL or path to the background reference (no subject).
        threshold: Luminance difference cutoff (0-255). Higher = less sensitive. Range 5-30.
        ramp: Soft transition width above threshold. Higher = softer edges.
        gamma: Gamma correction. Values < 1.0 brighten the output.
        blur_radius: Mask feathering as fraction of image size.
    """
    import wzrd

    gen_path = resolve_input(generated_image, suffix=".png")
    bg_path = resolve_input(background_image, suffix=".png")
    out_path = make_temp_path(suffix=".png")

    _creature_img, info = wzrd.subtract_background_file(
        generated_path=gen_path,
        background_path=bg_path,
        output_path=out_path,
        threshold=threshold,
        ramp=ramp,
        gamma=gamma,
        blur_radius=blur_radius,
    )

    return {
        "creature_image": upload(out_path),
        "info": {
            "mask_coverage": info.get("mask_coverage"),
            "creature_max_brightness": info.get("creature_max_brightness"),
            "creature_mean_brightness": info.get("creature_mean_brightness"),
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: subtract_background_video
# ---------------------------------------------------------------------------
@mcp.tool()
async def subtract_background_video(
    video: str,
    background_image: str,
    threshold: int = 10,
    gamma: float = 0.85,
    ramp: int = 20,
    blur_radius: float = 0.004,
    codec: str = "libx264",
    crf: int = 18,
    ctx: Optional[Context] = None,
) -> dict:
    """Remove background from an entire video frame-by-frame for projection mapping content.

    This is a long-running operation. Processing time scales with video length.

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
    import wzrd

    vid_path = resolve_input(video, suffix=".mp4")
    bg_path = resolve_input(background_image, suffix=".png")
    out_path = make_temp_path(suffix=".mp4")

    info = wzrd.subtract_background_video(
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

    return {
        "output_video": upload(out_path),
        "info": {
            "frames_processed": info.get("frames_processed"),
            "fps": info.get("fps"),
            "video_size": info.get("video_size"),
        },
    }


# ---------------------------------------------------------------------------
# Tool 3: detect_projection_surface
# ---------------------------------------------------------------------------
@mcp.tool()
async def detect_projection_surface(
    image: str,
    margin: float = 0.01,
    target_aspect_ratio: Optional[str] = None,
    output_resolution: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Detect and extract a projection surface from a photo.

    Finds the illuminated quadrilateral (screen/wall) and returns a rectified, perspective-corrected crop.

    Args:
        image: URL or path to photo containing the projection surface.
        margin: Inward shrinkage fraction (0.0–0.1) to avoid edge artifacts.
        target_aspect_ratio: Force output aspect ratio, e.g. "16:9" or "4:3".
        output_resolution: Output width in pixels. None keeps original resolution.
    """
    import wzrd

    img_path = resolve_input(image, suffix=".png")
    out_path = make_temp_path(suffix=".png")

    _cropped, info = wzrd.detect_projection_area(
        image_path=img_path,
        output_path=out_path,
        margin=margin,
        target_aspect_ratio=target_aspect_ratio,
        output_resolution=output_resolution,
    )

    return {
        "cropped_image": upload(out_path),
        "corners": info.get("corners"),
        "info": {
            "original_size": info.get("original_size"),
            "cropped_size": info.get("cropped_size"),
            "margin": info.get("margin"),
            "target_aspect_ratio": info.get("target_aspect_ratio"),
        },
    }


# ---------------------------------------------------------------------------
# Tool 4: align_images
# ---------------------------------------------------------------------------
@mcp.tool()
async def align_images(
    source_image: str,
    target_image: str,
    max_features: int = 10000,
    use_ecc_refinement: bool = True,
    ctx: Optional[Context] = None,
) -> dict:
    """Align a source image to match a target image's perspective.

    Uses SIFT/AKAZE feature matching with template matching fallback and optional ECC sub-pixel refinement.

    Args:
        source_image: URL or path to the image to warp.
        target_image: URL or path to the reference image to align to.
        max_features: SIFT feature detection limit.
        use_ecc_refinement: Enable sub-pixel ECC refinement for better accuracy.
    """
    import wzrd

    src_path = resolve_input(source_image, suffix=".png")
    tgt_path = resolve_input(target_image, suffix=".png")
    out_path = make_temp_path(suffix=".png")

    _warped, info = wzrd.align_images_file(
        source_path=src_path,
        target_path=tgt_path,
        output_path=out_path,
        max_features=max_features,
        use_ecc_refinement=use_ecc_refinement,
    )

    return {
        "aligned_image": upload(out_path),
        "info": {
            "confidence": info.get("confidence"),
            "method": info.get("method"),
            "num_inliers": info.get("num_inliers"),
            "num_matches": info.get("num_matches"),
        },
    }


# ---------------------------------------------------------------------------
# Tool 5: darken_surface
# ---------------------------------------------------------------------------
@mcp.tool()
async def darken_surface(
    image: str,
    max_brightness: float = 0.25,
    detail_boost: float = 1.15,
    target_aspect: str = "16:9",
    base_resolution: int = 1920,
    alignment_aids: bool = True,
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
        alignment_aids: Whether to generate an alignment aid video.
    """
    import wzrd

    img_path = resolve_input(image, suffix=".png")
    out_path = make_temp_path(suffix=".png")

    result = wzrd.darken_image_file(
        input_path=img_path,
        output_path=out_path,
        max_brightness=max_brightness,
        detail_boost=detail_boost,
        target_aspect=target_aspect,
        base_resolution=base_resolution,
        alignment_aids=alignment_aids,
    )

    pil_image = result["image"]
    pil_image.save(out_path)

    response: dict = {
        "darkened_image": upload(out_path),
        "alignment_video": None,
    }
    if result.get("video"):
        response["alignment_video"] = upload(str(result["video"]))

    return response


# ---------------------------------------------------------------------------
# Tool 6: prepare_surface
# ---------------------------------------------------------------------------
@mcp.tool()
async def prepare_surface(
    night_image: str,
    day_image: Optional[str] = None,
    target_aspect: str = "16:9",
    max_brightness: float = 0.20,
    margin: float = 0.01,
    alignment_aids: bool = True,
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
        alignment_aids: Generate alignment aid video.
    """
    import wzrd

    night_path = resolve_input(night_image, suffix=".png")
    day_path = resolve_input(day_image, suffix=".png") if day_image else None
    out_path = make_temp_path(suffix=".png")

    result = wzrd.prepare_surface(
        night_image_path=night_path,
        day_image_path=day_path,
        output_path=out_path,
        margin=margin,
        max_brightness=max_brightness,
        target_aspect=target_aspect,
        alignment_aids=alignment_aids,
    )

    pil_image = result["image"]
    pil_image.save(out_path)

    response: dict = {
        "surface_image": upload(out_path),
        "alignment_video": None,
    }
    if result.get("video"):
        response["alignment_video"] = upload(str(result["video"]))

    return response


# ---------------------------------------------------------------------------
# Tool 7: extract_color_regions
# ---------------------------------------------------------------------------
@mcp.tool()
async def extract_color_regions(
    image: str,
    max_colors: int = 5,
    min_area_fraction: float = 0.05,
    surface_image: Optional[str] = None,
    output_min_size: int = 512,
    ctx: Optional[Context] = None,
) -> dict:
    """Segment an image into color-based regions (islands) using K-means clustering.

    Useful for splitting a surface into zones for independent content projection.

    Args:
        image: URL or path to the input image.
        max_colors: Maximum number of color clusters.
        min_area_fraction: Minimum region size as fraction of total image area.
        surface_image: Optional surface image to extract corresponding region crops from.
        output_min_size: Minimum output dimension in pixels for each region crop.
    """
    import wzrd

    img_path = resolve_input(image, suffix=".png")
    surface_path = resolve_input(surface_image, suffix=".png") if surface_image else None
    out_dir = make_temp_dir()

    regions, output_dir = wzrd.extract_color_regions(
        image=img_path,
        output_dir=out_dir,
        max_colors=max_colors,
        min_area_fraction=min_area_fraction,
        surface=surface_path,
        output_min_size=output_min_size,
    )

    from pathlib import Path

    out = Path(output_dir)

    # Publish region files
    published_regions = []
    for region in regions:
        entry: dict = {"source_box": region.get("source_box")}
        if "filename" in region:
            entry["mask_path"] = upload(str(out / region["filename"]))
        if "surface_filename" in region:
            entry["surface_path"] = upload(str(out / region["surface_filename"]))
        published_regions.append(entry)

    # Publish metadata JSON
    metadata_file = out / "islands.json"
    metadata_url = upload(str(metadata_file)) if metadata_file.exists() else None

    return {
        "regions": published_regions,
        "metadata_path": metadata_url,
    }


# ---------------------------------------------------------------------------
# Tool 8: reproject_video
# ---------------------------------------------------------------------------
@mcp.tool()
async def reproject_video(
    video: str,
    island_metadata: str,
    target_aspect: str = "16:9",
    base_resolution: int = 1920,
    codec: str = "libx264",
    crf: int = 18,
    ctx: Optional[Context] = None,
) -> dict:
    """Place a processed video back at its original position on a canvas.

    Used to recompose island regions into a full-frame projection output for additive compositing.

    Args:
        video: URL or path to the island video.
        island_metadata: URL or path to the islands.json metadata file.
        target_aspect: Canvas aspect ratio, e.g. "16:9".
        base_resolution: Canvas width in pixels.
        codec: Video codec.
        crf: Quality setting — lower is better (0-51).
    """
    import wzrd

    vid_path = resolve_input(video, suffix=".mp4")
    meta_path = resolve_input(island_metadata, suffix=".json")
    out_path = make_temp_path(suffix=".mp4")

    info = wzrd.reproject_video_with_aspect(
        video_path=vid_path,
        island_metadata=meta_path,
        output_path=out_path,
        target_aspect=target_aspect,
        base_resolution=base_resolution,
        crf=crf,
        codec=codec,
    )

    return {
        "output_video": upload(out_path),
        "info": {
            "frames_processed": info.get("frames_processed"),
            "fps": info.get("fps"),
            "canvas_size": info.get("canvas_size"),
            "island_position": info.get("island_position"),
        },
    }
