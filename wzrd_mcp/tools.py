"""MCP tool wrappers that bridge MCP ↔ wzrd functions.

Each tool handles: resolve inputs (URL→local) → call wzrd → publish outputs.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from fastmcp import Context
from fastmcp.exceptions import ToolError

from .file_io import make_temp_dir, make_temp_path, resolve_input, upload
from ._log import log_call, log_progress, log_done, log_error
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
    t0 = log_call(_name, {
        "generated_image": generated_image, "background_image": background_image,
        "threshold": threshold, "ramp": ramp, "gamma": gamma, "blur_radius": blur_radius,
    })
    try:
        from wzrd.subtract_frame import subtract_background_file

        log_progress(_name, "Resolving inputs...")
        gen_path = resolve_input(generated_image, suffix=".png")
        bg_path = resolve_input(background_image, suffix=".png")
        out_path = make_temp_path(suffix=".png")

        log_progress(_name, "Running background subtraction...")
        _creature_img, info = subtract_background_file(
            generated_path=gen_path,
            background_path=bg_path,
            output_path=out_path,
            threshold=threshold,
            ramp=ramp,
            gamma=gamma,
            blur_radius=blur_radius,
        )

        log_progress(_name, "Uploading result...")
        result = {
            "creature_image": upload(out_path),
            "info": {
                "mask_coverage": info.get("mask_coverage"),
                "creature_max_brightness": info.get("creature_max_brightness"),
                "creature_mean_brightness": info.get("creature_mean_brightness"),
            },
        }
        log_done(_name, t0)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


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
    t0 = log_call(_name, {
        "video": video, "background_image": background_image,
        "threshold": threshold, "gamma": gamma, "ramp": ramp,
        "blur_radius": blur_radius, "codec": codec, "crf": crf,
    })
    try:
        from wzrd.subtract_video import subtract_background_video as _subtract_bg_video

        log_progress(_name, "Resolving inputs...")
        vid_path = resolve_input(video, suffix=".mp4")
        bg_path = resolve_input(background_image, suffix=".png")
        out_path = make_temp_path(suffix=".mp4")

        log_progress(_name, "Processing video frames...")
        info = _subtract_bg_video(
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

        log_progress(_name, "Uploading result...")
        result = {
            "output_video": upload(out_path),
            "info": {
                "frames_processed": info.get("frames_processed"),
                "fps": info.get("fps"),
                "video_size": info.get("video_size"),
            },
        }
        log_done(_name, t0)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 3: detect_projection_surface
# ---------------------------------------------------------------------------
@mcp.tool()
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
    t0 = log_call(_name, {
        "image": image, "margin": margin,
        "target_aspect_ratio": target_aspect_ratio, "output_resolution": output_resolution,
    })
    try:
        from wzrd.detect import detect_projection_area

        log_progress(_name, "Resolving input...")
        img_path = resolve_input(image, suffix=".png")
        out_path = make_temp_path(suffix=".png")

        log_progress(_name, "Detecting projection surface...")
        _cropped, info = detect_projection_area(
            image_path=img_path,
            output_path=out_path,
            margin=margin,
            target_aspect_ratio=target_aspect_ratio,
            output_resolution=output_resolution,
        )

        log_progress(_name, "Uploading result...")
        result = {
            "cropped_image": upload(out_path),
            "info": {
                "corners": info.get("corners"),
                "original_size": info.get("original_size"),
                "cropped_size": info.get("cropped_size"),
                "margin": info.get("margin"),
                "target_aspect_ratio": info.get("target_aspect_ratio"),
            },
        }
        log_done(_name, t0)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 4: align_images
# ---------------------------------------------------------------------------
@mcp.tool()
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
    t0 = log_call(_name, {
        "source_image": source_image, "target_image": target_image,
        "max_features": max_features,
    })
    try:
        from wzrd.align import align_images_file

        log_progress(_name, "Resolving inputs...")
        src_path = resolve_input(source_image, suffix=".png")
        tgt_path = resolve_input(target_image, suffix=".png")
        out_path = make_temp_path(suffix=".png")

        log_progress(_name, "Running feature matching & alignment...")
        _warped, info = align_images_file(
            source_path=src_path,
            target_path=tgt_path,
            output_path=out_path,
            max_features=max_features
        )

        log_progress(_name, "Uploading result...")
        result = {
            "aligned_image": upload(out_path),
            "info": {
                "confidence": info.get("confidence"),
                "method": info.get("method"),
                "num_inliers": info.get("num_inliers"),
                "num_matches": info.get("num_matches"),
            },
        }
        log_done(_name, t0)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


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
    t0 = log_call(_name, {
        "image": image, "max_brightness": max_brightness, "detail_boost": detail_boost,
        "target_aspect": target_aspect, "base_resolution": base_resolution,
        "alignment_aids": alignment_aids,
    })
    try:
        from wzrd.darken import darken_image_file

        log_progress(_name, "Resolving input...")
        img_path = resolve_input(image, suffix=".png")
        out_path = make_temp_path(suffix=".png")

        log_progress(_name, "Running darken pipeline...")
        result = darken_image_file(
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

        log_progress(_name, "Uploading result...")
        response: dict = {"darkened_image": upload(out_path)}
        if result.get("video"):
            response["alignment_video"] = upload(str(result["video"]))

        log_done(_name, t0)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 6: prepare_surface
# ---------------------------------------------------------------------------
@mcp.tool()
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
    t0 = log_call(_name, {
        "night_image": night_image, "day_image": day_image,
        "target_aspect": target_aspect, "max_brightness": max_brightness,
        "margin": margin, "alignment_aids": alignment_aids,
    })
    try:
        from wzrd.prepare_surface import prepare_surface as _prepare_surface

        log_progress(_name, "Resolving inputs...")
        night_path = resolve_input(night_image, suffix=".png")
        day_path = resolve_input(day_image, suffix=".png") if day_image != "" else None
        out_path = make_temp_path(suffix=".png")

        if day_path:
            log_progress(_name, "Running full pipeline: detect → align → darken...")
        else:
            log_progress(_name, "Running darken-only pipeline...")

        result = _prepare_surface(
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

        log_progress(_name, "Uploading result...")
        response: dict = {"surface_image": upload(out_path)}
        if result.get("video"):
            response["alignment_video"] = upload(str(result["video"]))

        log_done(_name, t0)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 7: extract_color_regions
# ---------------------------------------------------------------------------
@mcp.tool()
async def extract_color_regions(
    image: str,
    max_colors: int = 5,
    min_area_fraction: float = 0.05,
    surface_image: str = "",
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
    _name = "extract_color_regions"
    t0 = log_call(_name, {
        "image": image, "max_colors": max_colors, "min_area_fraction": min_area_fraction,
        "surface_image": surface_image, "output_min_size": output_min_size,
    })
    try:
        from wzrd.islands import extract_color_regions as _extract_color_regions

        log_progress(_name, "Resolving inputs...")
        img_path = resolve_input(image, suffix=".png")
        surface_path = resolve_input(surface_image, suffix=".png") if surface_image != "" else None
        out_dir = make_temp_dir()

        log_progress(_name, "Running K-means clustering...")
        regions, output_dir = _extract_color_regions(
            image=img_path,
            output_dir=out_dir,
            max_colors=max_colors,
            min_area_fraction=min_area_fraction,
            surface=surface_path,
            output_min_size=output_min_size,
        )

        from pathlib import Path

        out = Path(output_dir)

        log_progress(_name, f"Uploading {len(regions)} regions...")
        published_regions = []
        for region in regions:
            entry: dict = {"source_box": region.get("source_box")}
            if "filename" in region:
                entry["mask_path"] = upload(str(out / region["filename"]))
            if "surface_filename" in region:
                entry["surface_path"] = upload(str(out / region["surface_filename"]))
            published_regions.append(entry)

        metadata_file = out / "islands.json"
        metadata_url = upload(str(metadata_file)) if metadata_file.exists() else None

        result = {
            "regions": published_regions,
            "metadata_path": metadata_url,
        }
        log_done(_name, t0)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


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
    crf: int = 23,
    ctx: Optional[Context] = None,
) -> dict:
    """Place a processed video (typically a segmented section of a projection surface) back at its original position of the full projection canvas.

    Used to reposition generated island regions back into a full-frame projection output for additive compositing.

    Args:
        video: URL or path to the island video.
        island_metadata: URL or path to the islands.json metadata file.
        target_aspect: Canvas aspect ratio, e.g. "16:9".
        base_resolution: Canvas width in pixels.
        codec: Video codec.
        crf: Quality setting — lower is better (0-51).
    """
    _name = "reproject_video"
    t0 = log_call(_name, {
        "video": video, "island_metadata": island_metadata,
        "target_aspect": target_aspect, "base_resolution": base_resolution,
        "codec": codec, "crf": crf,
    })
    try:
        from wzrd.reproject import reproject_video_with_aspect

        log_progress(_name, "Resolving inputs...")
        vid_path = resolve_input(video, suffix=".mp4")
        meta_path = resolve_input(island_metadata, suffix=".json")
        out_path = make_temp_path(suffix=".mp4")

        log_progress(_name, "Reprojecting video onto canvas...")
        info = reproject_video_with_aspect(
            video_path=vid_path,
            island_metadata=meta_path,
            output_path=out_path,
            target_aspect=target_aspect,
            base_resolution=base_resolution,
            crf=crf,
            codec=codec,
        )

        log_progress(_name, "Uploading result...")
        result = {
            "output_video": upload(out_path),
            "info": {
                "frames_processed": info.get("frames_processed"),
                "fps": info.get("fps"),
                "canvas_size": info.get("canvas_size"),
                "island_position": info.get("island_position"),
            },
        }
        log_done(_name, t0)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))


# ---------------------------------------------------------------------------
# Tool 9: texture_flow
# ---------------------------------------------------------------------------

# S3/CloudFront settings for downloading TextureFlow outputs
_TF_BUCKET = os.getenv("AWS_BUCKET_NAME", "edenartlab-stage-data")
_TF_REGION = os.getenv("AWS_REGION_NAME", "us-east-1")
_TF_CLOUDFRONT = os.getenv("CLOUDFRONT_URL")


def _tf_file_url(filename: str) -> str:
    """Build a download URL for an S3 key from the TextureFlow result."""
    if _TF_CLOUDFRONT:
        return f"{_TF_CLOUDFRONT}/{filename}"
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


@mcp.tool()
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
        base_model: SD1.5 checkpoint. Options: "SD15/juggernaut_reborn.safetensors" (realistic, best default), "SD15/darkSushiMixMix_225D.safetensors" (creative/abstract, pastel colors), "SD15/protogenV22Anime_protogenV22.safetensors" (anime/cartoon, flat colors).
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
    t0 = log_call(_name, {
        "images": images, "n_seconds": n_seconds, "width": width, "height": height,
        "base_model": base_model, "use_controlnet1": use_controlnet1,
        "mapping_mode": mapping_mode, "use_upscale": use_upscale, "seed": seed,
    })
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
        result = instance.run.remote(tool_key="texture_flow", args=args)

        log_progress(_name, "Extracting output URLs...")
        output_urls = _tf_extract_urls(result)

        response = {
            "output_videos": [url for url, _fname in output_urls],
        }
        log_done(_name, t0)
        return response
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))
