"""
Simulate the real-world viewing experience of additive projection mapping.

Given a projector output video (purely additive light on black) and a
projection surface image, composites them to predict what a human viewer
would see: the surface texture plus projected light modulated by surface
reflectivity.

Physics model:
    output_pixel = surface + projector_frame * reflectivity

Where reflectivity is derived from the surface image's luminance — dark/black
regions (sky, non-reflective areas) absorb projected light, while brighter
surface regions reflect it back to the viewer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Union, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import VideoWriter

# Defaults
DEFAULT_SURFACE_WEIGHT = 0.35
DEFAULT_PROJECTION_STRENGTH = 1.0
DEFAULT_AMBIENT = 0.0
DEFAULT_CRF = 18
DEFAULT_CODEC = "libx264"


def _get_video_info(video_path: Union[str, Path]) -> Tuple[int, int, float, Optional[int]]:
    """Get video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split(",")
    width, height = int(parts[0]), int(parts[1])
    fps_num, fps_den = map(int, parts[2].split("/"))
    fps = fps_num / fps_den
    try:
        frame_count = int(parts[3])
    except (ValueError, IndexError):
        frame_count = None
    return width, height, fps, frame_count


def _iter_video_frames(
    video_path: Union[str, Path],
    width: int,
    height: int,
):
    """Stream video frames as uint8 numpy arrays via ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "error",
        "-",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height * 3
    frame_num = 0
    try:
        while True:
            raw = process.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame_num += 1
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            yield frame_num, arr
    finally:
        process.stdout.close()
        process.wait()


def compute_reflectivity(
    surface_arr: np.ndarray,
    ambient: float = DEFAULT_AMBIENT,
) -> np.ndarray:
    """Derive a per-pixel reflectivity map from the surface image.

    Uses luminance of the surface as a proxy for reflectivity.
    Pure black regions (e.g. masked-out sky) get zero reflectivity.

    Args:
        surface_arr: Surface image as float32 HWC, range [0, 255].
        ambient: Minimum reflectivity floor (0.0-1.0). Adds a small
                 baseline so even dark surfaces reflect a tiny bit.

    Returns:
        Float32 HW array in [0, 1] representing reflectivity per pixel.
    """
    luminance = (
        0.299 * surface_arr[:, :, 0]
        + 0.587 * surface_arr[:, :, 1]
        + 0.114 * surface_arr[:, :, 2]
    )
    reflectivity = luminance / 255.0
    if ambient > 0:
        # Only add ambient to non-zero pixels (truly black stays black)
        nonzero = luminance > 1.0
        reflectivity[nonzero] = np.clip(reflectivity[nonzero] + ambient, 0, 1)
    return reflectivity


def composite_frame(
    projector_frame: np.ndarray,
    surface_arr: np.ndarray,
    reflectivity: np.ndarray,
    surface_weight: float = DEFAULT_SURFACE_WEIGHT,
    projection_strength: float = DEFAULT_PROJECTION_STRENGTH,
) -> np.ndarray:
    """Composite a single projector frame onto the surface.

    Args:
        projector_frame: Projector output as float32 HWC [0, 255].
        surface_arr: Surface image as float32 HWC [0, 255].
        reflectivity: Per-pixel reflectivity HW float32 [0, 1].
        surface_weight: How visible the underlying surface texture is (0-1).
                        0 = only projected light, 1 = full surface + projection.
        projection_strength: Multiplier for projected light intensity.

    Returns:
        Composited frame as uint8 HWC.
    """
    # Surface contribution (ambient appearance of the wall/surface)
    base = surface_arr * surface_weight

    # Projected light modulated by surface reflectivity
    # Expand reflectivity to 3 channels
    refl_3ch = reflectivity[:, :, np.newaxis]
    projected = projector_frame * refl_3ch * projection_strength

    result = np.clip(base + projected, 0, 255).astype(np.uint8)
    return result


def simulate_view(
    projector_video_path: Union[str, Path],
    surface_image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    surface_weight: float = DEFAULT_SURFACE_WEIGHT,
    projection_strength: float = DEFAULT_PROJECTION_STRENGTH,
    ambient: float = DEFAULT_AMBIENT,
    crf: int = DEFAULT_CRF,
    codec: str = DEFAULT_CODEC,
) -> dict:
    """Simulate the real-world viewing experience of projection mapping.

    Composites a projector output video (purely additive light on black
    background) onto a projection surface image, producing a video that
    approximates what a human viewer would see.

    Args:
        projector_video_path: Path to the projector output video (additive
            light content — background-subtracted kling video or raw
            textureflow output).
        surface_image_path: Path to the projection surface image. Ideally
            pre-processed with non-reflective regions (sky etc.) turned to
            pure black.
        output_path: Output video path (auto-generated if None).
        surface_weight: Visibility of underlying surface texture (0.0-1.0).
            Lower = more contrast from projection, higher = more visible
            surface. Default 0.35.
        projection_strength: Multiplier for projected light (0.5-2.0).
            Values > 1.0 simulate a brighter projector. Default 1.0.
        ambient: Minimum reflectivity floor for non-black surface pixels
            (0.0-0.3). Adds subtle light bounce even on dark surfaces.
        crf: Video quality (0-51, lower = better).
        codec: Video codec.

    Returns:
        Info dict with output path and processing metadata.
    """
    projector_video_path = Path(projector_video_path)
    surface_image_path = Path(surface_image_path)

    # Get video dimensions
    vid_w, vid_h, fps, frame_count = _get_video_info(projector_video_path)

    # Ensure even dimensions
    out_w = vid_w - (vid_w % 2)
    out_h = vid_h - (vid_h % 2)

    # Load and resize surface to match video
    surface = Image.open(surface_image_path).convert("RGB")
    if surface.size != (out_w, out_h):
        surface = surface.resize((out_w, out_h), Image.Resampling.LANCZOS)
    surface_arr = np.array(surface, dtype=np.float32)

    # Precompute reflectivity map
    reflectivity = compute_reflectivity(surface_arr, ambient=ambient)

    # Setup output
    if output_path is None:
        output_path = projector_video_path.with_name(
            f"{projector_video_path.stem}_simulated_view.mp4"
        )
    output_path = Path(output_path)

    # Process frames
    frame_num = 0
    with VideoWriter(output_path, out_w, out_h, fps, crf=crf, codec=codec) as writer:
        for fn, frame_u8 in _iter_video_frames(projector_video_path, vid_w, vid_h):
            frame_num = fn

            # Crop to even dimensions if needed
            frame_u8 = frame_u8[:out_h, :out_w]
            frame_f = frame_u8.astype(np.float32)

            composited = composite_frame(
                frame_f, surface_arr, reflectivity,
                surface_weight=surface_weight,
                projection_strength=projection_strength,
            )
            writer.write(composited)

    return {
        "output_path": str(output_path),
        "video_size": (out_w, out_h),
        "fps": fps,
        "frames_processed": frame_num,
        "frame_count": frame_count,
        "surface_weight": surface_weight,
        "projection_strength": projection_strength,
        "ambient": ambient,
    }


# ---------------------------------------------------------------------------
# CLI: python -m wzrd.simulate_view
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate real-world projection mapping viewing experience.",
    )
    parser.add_argument("projector_video", help="Path to projector output video")
    parser.add_argument("surface_image", help="Path to projection surface image")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument("--surface-weight", type=float, default=DEFAULT_SURFACE_WEIGHT)
    parser.add_argument("--projection-strength", type=float, default=DEFAULT_PROJECTION_STRENGTH)
    parser.add_argument("--ambient", type=float, default=DEFAULT_AMBIENT)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument("--codec", default=DEFAULT_CODEC)
    args = parser.parse_args()

    print(f"Projector video: {args.projector_video}")
    print(f"Surface image:   {args.surface_image}")

    info = simulate_view(
        args.projector_video,
        args.surface_image,
        output_path=args.output,
        surface_weight=args.surface_weight,
        projection_strength=args.projection_strength,
        ambient=args.ambient,
        crf=args.crf,
        codec=args.codec,
    )

    print(f"\nDone!")
    print(f"  Output: {info['output_path']}")
    print(f"  Frames: {info['frames_processed']}")
    print(f"  Size:   {info['video_size'][0]}x{info['video_size'][1]} @ {info['fps']:.1f}fps")


if __name__ == "__main__":
    _cli()
