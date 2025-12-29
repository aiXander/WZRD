"""
Video processing for background subtraction.

Functions to process video files frame-by-frame, applying background
subtraction to extract creatures/elements across all frames.
"""

from PIL import Image
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Union, Optional, Tuple, Literal, Callable, Iterator
import cv2

from .utils import align_to_reference, get_aspect_ratio, aspect_ratios_match
from .subtract import compute_difference_mask, extract_creature
from .config import get_subtract_config, get_resolution_config


DiffMode = Literal['rgb', 'lab', 'luminance']
OutputMode = Literal['additive', 'alpha']


def get_video_info(video_path: Union[str, Path]) -> Tuple[int, int, float, Optional[int]]:
    """
    Get video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height, fps, frame_count)
        frame_count may be None for some formats
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split(',')
    width, height = int(parts[0]), int(parts[1])
    fps_num, fps_den = map(int, parts[2].split('/'))
    fps = fps_num / fps_den
    try:
        frame_count = int(parts[3])
    except (ValueError, IndexError):
        frame_count = None
    return width, height, fps, frame_count


def process_frame(
    gen_arr: np.ndarray,
    bg_arr: np.ndarray,
    threshold: int,
    boost: float,
    feather_radius: int,
    diff_mode: DiffMode,
    min_alpha: float,
    use_opencv_blur: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single frame using background subtraction.

    Args:
        gen_arr: Generated frame as float32 array (HWC, 0-255)
        bg_arr: Background frame as float32 array (HWC, 0-255)
        threshold: Difference threshold
        boost: Brightness boost multiplier
        feather_radius: Edge feather radius
        diff_mode: Difference calculation mode
        min_alpha: Minimum mask value
        use_opencv_blur: Use OpenCV (faster) instead of PIL for blur

    Returns:
        Tuple of (creature array, mask array)
    """
    if use_opencv_blur and feather_radius > 0:
        # Use OpenCV for faster Gaussian blur
        mask = _compute_mask_opencv(
            gen_arr, bg_arr, threshold, feather_radius, diff_mode, min_alpha
        )
    else:
        mask = compute_difference_mask(
            gen_arr, bg_arr, threshold, feather_radius, diff_mode, min_alpha
        )

    creature = extract_creature(gen_arr, bg_arr, mask, boost)
    return creature, mask


def _compute_mask_opencv(
    generated: np.ndarray,
    background: np.ndarray,
    threshold: int,
    feather_radius: int,
    diff_mode: DiffMode,
    min_alpha: float,
) -> np.ndarray:
    """Compute mask using OpenCV for faster Gaussian blur."""
    if diff_mode == 'rgb':
        diff = np.abs(generated - background)
        diff_magnitude = np.max(diff, axis=2)

    elif diff_mode == 'lab':
        try:
            from skimage import color
        except ImportError:
            raise ImportError("scikit-image required for LAB mode")
        gen_lab = color.rgb2lab(generated / 255.0)
        bg_lab = color.rgb2lab(background / 255.0)
        diff_magnitude = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))
        diff_magnitude = np.clip(diff_magnitude * 2.55, 0, 255)

    elif diff_mode == 'luminance':
        gen_lum = 0.299 * generated[:, :, 0] + 0.587 * generated[:, :, 1] + 0.114 * generated[:, :, 2]
        bg_lum = 0.299 * background[:, :, 0] + 0.587 * background[:, :, 1] + 0.114 * background[:, :, 2]
        diff_magnitude = np.abs(gen_lum - bg_lum)
    else:
        raise ValueError(f"Unknown diff_mode: {diff_mode}")

    mask = (diff_magnitude > threshold).astype(np.float32)

    # Use OpenCV GaussianBlur (much faster than PIL)
    if feather_radius > 0:
        kernel_size = feather_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return np.clip(mask, min_alpha, 1.0)


def iter_video_frames(
    video_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Iterate over video frames using streaming FFmpeg pipe.

    Args:
        video_path: Path to video file
        target_size: Optional (width, height) to resize frames

    Yields:
        Tuple of (frame_number, frame_array)
    """
    width, height, fps, _ = get_video_info(video_path)

    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-v', 'error',
        '-'
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height * 3
    frame_num = 0

    try:
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) < frame_size:
                break

            frame_num += 1
            frame_arr = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

            if target_size and (width, height) != target_size:
                frame_pil = Image.fromarray(frame_arr)
                frame_pil = frame_pil.resize(target_size, Image.Resampling.LANCZOS)
                frame_arr = np.array(frame_pil)

            yield frame_num, frame_arr
    finally:
        process.stdout.close()
        process.wait()


def subtract_background_video(
    video_path: Union[str, Path],
    background_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    threshold: Optional[int] = None,
    boost: Optional[float] = None,
    feather_radius: Optional[int] = None,
    diff_mode: Optional[DiffMode] = None,
    output_mode: Optional[OutputMode] = None,
    preview: bool = False,
    crf: int = 18,
    codec: str = 'libx264',
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Process a video file with background subtraction.

    Uses streaming FFmpeg pipes for memory efficiency.

    Args:
        video_path: Path to input video
        background_path: Path to background frame image
        output_path: Path for output video (auto-generated if None)
        threshold: Difference threshold (default from config)
        boost: Brightness boost (default from config)
        feather_radius: Edge feather radius (default from config)
        diff_mode: Difference calculation mode (default from config)
        output_mode: 'additive' or 'alpha' (default from config)
        preview: Also save a preview video with composite
        crf: FFmpeg CRF quality (0-51, lower=better)
        codec: Video codec (default: libx264)
        progress_callback: Optional callback(frame_num, total_frames)

    Returns:
        Info dict with processing details
    """
    video_path = Path(video_path)
    background_path = Path(background_path)

    # Load config defaults
    config = get_subtract_config()
    resolution_config = get_resolution_config()

    threshold = threshold if threshold is not None else config['threshold']
    boost = boost if boost is not None else config['boost']
    feather_radius = feather_radius if feather_radius is not None else config['feather_radius']
    diff_mode = diff_mode if diff_mode is not None else config['diff_mode']
    output_mode = output_mode if output_mode is not None else config['output_mode']
    min_alpha = config['min_alpha']
    aspect_tolerance = resolution_config['aspect_tolerance']

    # Load background
    background = Image.open(background_path).convert('RGB')
    bg_size = background.size

    # Ensure even dimensions for codec compatibility
    output_width = bg_size[0] if bg_size[0] % 2 == 0 else bg_size[0] - 1
    output_height = bg_size[1] if bg_size[1] % 2 == 0 else bg_size[1] - 1
    if (output_width, output_height) != bg_size:
        background = background.crop((0, 0, output_width, output_height))
        bg_size = (output_width, output_height)

    bg_arr = np.array(background, dtype=np.float32)

    # Get video info
    width, height, fps, frame_count = get_video_info(video_path)
    video_aspect = width / height
    bg_aspect = get_aspect_ratio(bg_size)
    needs_alignment = (width, height) != bg_size

    # Setup output paths
    if output_path is None:
        params_str = f"t{threshold}_b{boost:.1f}_f{feather_radius}_{diff_mode}"
        if output_mode == 'alpha':
            output_path = video_path.with_name(f"{video_path.stem}_creature_{params_str}_alpha.mov")
        else:
            output_path = video_path.with_name(f"{video_path.stem}_creature_{params_str}.mp4")
    output_path = Path(output_path)

    preview_path = output_path.with_name(f"{output_path.stem}_preview.mp4") if preview else None

    info = {
        'input_video': str(video_path),
        'output_video': str(output_path),
        'background': str(background_path),
        'video_size': (width, height),
        'output_size': bg_size,
        'fps': fps,
        'frame_count': frame_count,
        'threshold': threshold,
        'boost': boost,
        'feather_radius': feather_radius,
        'diff_mode': diff_mode,
        'output_mode': output_mode,
        'needs_alignment': needs_alignment,
    }

    # Setup output format
    if output_mode == 'alpha':
        output_pix_fmt_in = 'rgba'
        pix_fmt_out = 'yuva444p10le'
        output_codec = 'prores_ks'
        extra_args = ['-profile:v', '4444']
    else:
        output_pix_fmt_in = 'rgb24'
        pix_fmt_out = 'yuv420p'
        output_codec = codec
        extra_args = ['-crf', str(crf)]

    # Setup FFmpeg output pipe
    output_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{bg_size[0]}x{bg_size[1]}',
        '-pix_fmt', output_pix_fmt_in,
        '-r', str(fps),
        '-i', '-',
        '-c:v', output_codec,
        *extra_args,
        '-pix_fmt', pix_fmt_out,
        '-v', 'error',
        str(output_path)
    ]

    preview_cmd = None
    if preview:
        preview_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{bg_size[0]}x{bg_size[1]}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', codec,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-v', 'error',
            str(preview_path)
        ]

    # Start output processes
    output_process = subprocess.Popen(output_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    preview_process = subprocess.Popen(preview_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) if preview_cmd else None

    frame_num = 0
    try:
        for frame_num, frame_arr in iter_video_frames(video_path):
            if progress_callback:
                progress_callback(frame_num, frame_count or 0)

            # Align if needed
            if needs_alignment:
                frame_pil = Image.fromarray(frame_arr)
                frame_pil, _ = align_to_reference(frame_pil, background, aspect_tolerance)
                frame_arr = np.array(frame_pil)

            gen_arr = frame_arr.astype(np.float32)

            # Process frame
            creature, mask = process_frame(
                gen_arr, bg_arr, threshold, boost, feather_radius,
                diff_mode, min_alpha, use_opencv_blur=True
            )

            # Write to output
            if output_mode == 'alpha':
                alpha_channel = (mask * 255).astype(np.uint8)
                rgba = np.dstack([creature, alpha_channel])
                output_process.stdin.write(rgba.tobytes())
            else:
                output_process.stdin.write(creature.tobytes())

            # Write preview
            if preview_process:
                composite = np.clip(bg_arr + creature.astype(np.float32), 0, 255).astype(np.uint8)
                preview_process.stdin.write(composite.tobytes())

        info['frames_processed'] = frame_num

    finally:
        output_process.stdin.close()
        output_process.wait()
        if preview_process:
            preview_process.stdin.close()
            preview_process.wait()

    if preview_path:
        info['preview_video'] = str(preview_path)

    return info
