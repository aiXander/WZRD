"""
Video processing for background subtraction.

Functions to process video files frame-by-frame, applying background
subtraction to extract creatures/elements across all frames.
"""

import time
from PIL import Image
import numpy as np
import subprocess
from pathlib import Path
from typing import Union, Optional, Tuple, Literal, Callable, Iterator, List
import cv2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from queue import Queue
from threading import Thread

from .utils import (
    get_aspect_ratio, aspect_ratios_match,
    extract_creature,
    downscale_arr, upscale_arr,
    frac_to_px, VideoWriter,
)

# Module-level defaults
DEFAULT_THRESHOLD = 10
DEFAULT_RAMP = 20
DEFAULT_GAMMA = 0.85
DEFAULT_BLUR_RADIUS = 0.004         # fraction of min(H,W); ≈ 6 px at 1080p
DEFAULT_OUTPUT_MODE = 'additive'
DEFAULT_MIN_ALPHA = 0.0
DEFAULT_MORPH_SIZE = 0               # off by default (0 = disabled)
DEFAULT_ASPECT_TOLERANCE = 0.02
DEFAULT_COLOR_CORRECTION_THRESHOLD = 15  # max luminance diff to count as "background" pixel

# Number of frames to subsample when estimating the global color shift
COLOR_SHIFT_SAMPLE_FRAMES = 50

OutputMode = Literal['additive', 'alpha']


def get_video_info(
    video_path: Union[str, Path],
) -> Tuple[int, int, float, Optional[int]]:
    """
    Get video metadata using ffprobe.

    Args:
        video_path: Path to video file.

    Returns:
        Tuple of (width, height, fps, frame_count).
        frame_count may be None for some formats.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'csv=p=0',
        str(video_path),
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


def _luminance(arr: np.ndarray) -> np.ndarray:
    """Compute luminance from float32 RGB (HWC, 0-255). Returns HW float32."""
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def estimate_color_shift(
    video_path: Union[str, Path],
    bg_arr: np.ndarray,
    output_size: Tuple[int, int],
    threshold: float = DEFAULT_COLOR_CORRECTION_THRESHOLD,
    n_samples: int = COLOR_SHIFT_SAMPLE_FRAMES,
) -> np.ndarray:
    """Estimate a global color shift between the AI video and the background.

    Subsamples *n_samples* frames, finds the one with the lowest overall
    luminance difference to the background (i.e. the most "static" frame),
    then computes the per-channel RGB shift using only pixels whose
    luminance difference is below *threshold*.

    Args:
        video_path:   Path to the AI-generated video.
        bg_arr:       Background image as float32 (HWC, 0-255).
        output_size:  (width, height) to resize frames to.
        threshold:    Max luminance difference for a pixel to count as
            "background" when computing the shift.
        n_samples:    Number of frames to subsample.

    Returns:
        3-element float32 RGB shift vector (to be added to bg_arr).
    """
    _, _, _, frame_count = get_video_info(video_path)
    if frame_count is None or frame_count == 0:
        return np.zeros(3, dtype=np.float32)

    # Pick evenly-spaced frame indices
    step = max(1, frame_count // n_samples)
    sample_indices = set(range(0, frame_count, step))

    bg_lum = _luminance(bg_arr)

    best_frame = None
    best_diff = float('inf')

    for fn, frame_u8 in iter_video_frames(video_path, target_size=output_size):
        if fn not in sample_indices:
            continue
        frame_f = frame_u8.astype(np.float32)
        # Crop to output size if needed (odd dimension handling)
        frame_f = frame_f[:bg_arr.shape[0], :bg_arr.shape[1]]
        lum_diff = np.abs(_luminance(frame_f) - bg_lum)
        mean_diff = float(np.mean(lum_diff))
        if mean_diff < best_diff:
            best_diff = mean_diff
            best_frame = frame_f

    if best_frame is None:
        return np.zeros(3, dtype=np.float32)

    # Build mask of "confident background" pixels (low luminance diff)
    lum_diff = np.abs(_luminance(best_frame) - bg_lum)
    bg_mask = lum_diff < threshold

    if np.sum(bg_mask) < 100:
        return np.zeros(3, dtype=np.float32)

    # Per-channel RGB shift (video_frame - background)
    shift = np.array([
        np.mean(best_frame[bg_mask, 0] - bg_arr[bg_mask, 0]),
        np.mean(best_frame[bg_mask, 1] - bg_arr[bg_mask, 1]),
        np.mean(best_frame[bg_mask, 2] - bg_arr[bg_mask, 2]),
    ], dtype=np.float32)

    return shift


def iter_video_frames(
    video_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Iterate over video frames using streaming FFmpeg pipe.

    Args:
        video_path:  Path to video file.
        target_size: Optional (width, height) to resize frames.

    Yields:
        Tuple of (frame_number, frame_array).
    """
    width, height, fps, _ = get_video_info(video_path)

    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-v', 'error',
        '-',
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    frame_size = width * height * 3
    frame_num = 0

    try:
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) < frame_size:
                break

            frame_num += 1
            frame_arr = np.frombuffer(
                raw_frame, dtype=np.uint8
            ).reshape((height, width, 3))

            if target_size and (width, height) != target_size:
                frame_arr = cv2.resize(
                    frame_arr, target_size,
                    interpolation=cv2.INTER_LANCZOS4,
                )

            yield frame_num, frame_arr
    finally:
        process.stdout.close()
        process.wait()


def iter_video_frames_batched(
    video_path: Union[str, Path],
    batch_size: int = 16,
    target_size: Optional[Tuple[int, int]] = None,
) -> Iterator[List[Tuple[int, np.ndarray]]]:
    """
    Iterate over video frames in batches.

    Args:
        video_path:  Path to video file.
        batch_size:  Number of frames per batch.
        target_size: Optional (width, height) to resize frames.

    Yields:
        List of (frame_number, frame_array) tuples.
    """
    batch = []
    for frame_num, frame_arr in iter_video_frames(video_path, target_size):
        batch.append((frame_num, frame_arr))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def subtract_background_video(
    video_path: Union[str, Path],
    background_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    threshold: int = DEFAULT_THRESHOLD,
    ramp: int = DEFAULT_RAMP,
    gamma: float = DEFAULT_GAMMA,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    output_mode: OutputMode = DEFAULT_OUTPUT_MODE,
    morph_size: float = DEFAULT_MORPH_SIZE,
    color_correction: bool = True,
    color_correction_threshold: float = DEFAULT_COLOR_CORRECTION_THRESHOLD,
    subtract_bg: bool = True,
    preview: bool = False,
    crf: int = 18,
    codec: str = 'libx264',
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Process a video file with background subtraction.

    Pipeline per frame:
      1. Luminance difference against (color-corrected) background
      2. Soft ramp threshold → mask
      3. Optional morphological opening
      4. Gaussian blur for smooth falloff
      5. Background subtraction with smooth mask modulation
      6. Gamma correction + mask application

    Color correction is computed **once** before processing: subsample
    frames, find the most static one, estimate a global RGB shift from
    low-diff pixels, and adjust the background to match.

    Args:
        video_path:  Path to input video.
        background_path: Path to background frame image.
        output_path: Path for output video (auto-generated if None).
        threshold:   Luminance difference threshold (0-255).
        ramp:        Soft ramp width above threshold.
        gamma:       Gamma correction (< 1.0 brightens).
        blur_radius: Gaussian blur radius for mask feathering, as fraction
            of min(H,W).  0 = disable.
        output_mode: ``'additive'`` or ``'alpha'``.
        morph_size:  Morphological kernel size as fraction of min(H,W)
            (0 = disable, default).
        color_correction: Estimate and correct global color shift.
        color_correction_threshold: Max luminance diff for "background"
            pixels during color shift estimation.
        subtract_bg: Subtract background before masking (default True).
        preview:     Also save a preview video with composite.
        crf:         FFmpeg CRF quality (0-51, lower = better).
        codec:       Video codec (default: libx264).
        progress_callback: Optional callback(frame_num, total_frames).

    Returns:
        Info dict with processing details.
    """
    video_path = Path(video_path)
    background_path = Path(background_path)
    min_alpha = DEFAULT_MIN_ALPHA

    # Get video info first — resize the background once instead of every frame
    width, height, fps, frame_count = get_video_info(video_path)

    # Ensure even dimensions for codec compatibility
    output_width = width - (width % 2)
    output_height = height - (height % 2)
    output_size = (output_width, output_height)

    # Load background and resize to match video dimensions
    background = Image.open(background_path).convert('RGB')
    bg_orig_size = background.size
    if background.size != output_size:
        background = background.resize(output_size, Image.Resampling.LANCZOS)

    bg_arr = np.array(background, dtype=np.float32)

    # ---- One-time color correction ------------------------------------------
    color_shift = np.zeros(3, dtype=np.float32)
    if color_correction:
        print('  Estimating color shift...', flush=True)
        color_shift = estimate_color_shift(
            video_path, bg_arr, output_size,
            threshold=color_correction_threshold,
        )
        if np.any(np.abs(color_shift) > 0.5):
            print(f'  Color shift (RGB): [{color_shift[0]:+.1f}, '
                  f'{color_shift[1]:+.1f}, {color_shift[2]:+.1f}]', flush=True)
            # Shift the background to match the AI video's color space
            bg_arr = np.clip(bg_arr + color_shift, 0, 255)
        else:
            print('  No significant color shift detected.', flush=True)
            color_shift = np.zeros(3, dtype=np.float32)

    # Precompute background luminance at half res
    bg_arr_half = downscale_arr(bg_arr, 2)
    bg_lum_half = _luminance(bg_arr_half)

    # Only need to crop 1px if video has odd dimensions
    _needs_crop = (width, height) != output_size

    # Setup output paths
    if output_path is None:
        params_str = f"t{threshold}_g{gamma:.2f}_r{ramp}"
        if output_mode == 'alpha':
            output_path = video_path.with_name(
                f"{video_path.stem}_creature_{params_str}_alpha.mov"
            )
        else:
            output_path = video_path.with_name(
                f"{video_path.stem}_creature_{params_str}.mp4"
            )
    output_path = Path(output_path)

    preview_path = (
        output_path.with_name(f"{output_path.stem}_preview.mp4")
        if preview else None
    )

    info = {
        'input_video': str(video_path),
        'output_video': str(output_path),
        'background': str(background_path),
        'video_size': (width, height),
        'output_size': output_size,
        'fps': fps,
        'frame_count': frame_count,
        'threshold': threshold,
        'ramp': ramp,
        'gamma': gamma,
        'blur_radius': blur_radius,
        'output_mode': output_mode,
        'morph_size': morph_size,
        'color_shift': color_shift.tolist(),
        'bg_original_size': bg_orig_size,
    }

    # ---- Video writers -------------------------------------------------------
    is_alpha = output_mode == 'alpha'
    output_writer = VideoWriter(
        output_path, output_size[0], output_size[1], fps,
        alpha=is_alpha, crf=crf, codec=codec,
    )
    preview_writer = (
        VideoWriter(preview_path, output_size[0], output_size[1], fps, crf=crf, codec=codec)
        if preview else None
    )

    # ---- Parallel batch processing ------------------------------------------
    num_workers = min(cpu_count(), 8)
    batch_size = num_workers * 2

    # Convert fractional sizes to pixels at half resolution
    half_ref = min(output_height // 2, output_width // 2)
    morph_px = frac_to_px(morph_size, half_ref, odd=True) if morph_size > 0 else 0
    blur_px = frac_to_px(blur_radius, half_ref) if blur_radius > 0 else 0
    # Gaussian kernel size must be odd
    blur_ksize = blur_px * 2 + 1 if blur_px > 0 else 0

    def _compute_frame_data(args):
        """Thread worker: compute luminance diff mask at half res."""
        fn, frame_arr = args

        # Crop 1px if video has odd dimensions
        if _needs_crop:
            frame_arr = frame_arr[:output_height, :output_width]

        gen_arr = frame_arr.astype(np.float32)
        gen_arr_half = downscale_arr(gen_arr, 2)

        # --- Luminance difference at half res --------------------------------
        gen_lum_half = _luminance(gen_arr_half)
        diff_mag = np.abs(gen_lum_half - bg_lum_half)

        # Soft ramp threshold
        ramp_width = max(float(ramp), 1e-6)
        mask = np.clip(
            (diff_mag - threshold) / ramp_width, 0, 1,
        ).astype(np.float32)

        # Morphological opening (half res, off by default)
        if morph_px > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morph_px, morph_px),
            )
            mask_u8 = (mask * 255).astype(np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kern)
            mask = mask_u8.astype(np.float32) / 255.0

        # --- Gaussian blur for smooth falloff (half res) ---------------------
        # Use max(original, blurred) so the blur only *adds* glow around
        # edges — small bright features keep their full mask value.
        if blur_ksize > 0:
            blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
            mask = np.maximum(mask, blurred)

        # --- Upscale mask to full res ---------------------------------------
        mask = upscale_arr(mask, (output_height, output_width))

        return fn, gen_arr, mask

    # Prefetch frames in a background thread to overlap I/O with compute
    frame_queue: Queue = Queue(maxsize=2)

    def _read_frames():
        for batch in iter_video_frames_batched(
            video_path, batch_size=batch_size
        ):
            frame_queue.put(batch)
        frame_queue.put(None)

    reader = Thread(target=_read_frames, daemon=True)
    reader.start()

    frame_num = 0

    writers = [output_writer]
    if preview_writer:
        writers.append(preview_writer)

    # Enter all writer context managers
    for w in writers:
        w.__enter__()

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while True:
                batch = frame_queue.get()
                if batch is None:
                    break

                # Parallel: mask computation
                results = list(
                    executor.map(_compute_frame_data, batch)
                )
                results.sort(key=lambda x: x[0])

                # Sequential: creature extraction → write
                for fn, gen_arr, mask in results:
                    frame_num = fn

                    # Apply min_alpha
                    if min_alpha > 0:
                        mask = np.clip(mask, min_alpha, 1.0)

                    # Extract creature
                    creature = extract_creature(
                        gen_arr, bg_arr, mask, gamma=gamma,
                        subtract_bg=subtract_bg,
                    )

                    if progress_callback:
                        progress_callback(frame_num, frame_count or 0)

                    # Write to output
                    if output_mode == 'alpha':
                        alpha_channel = (mask * 255).astype(np.uint8)
                        rgba = np.dstack([creature, alpha_channel])
                        output_writer.write(rgba)
                    else:
                        output_writer.write(creature)

                    # Write preview
                    if preview_writer:
                        composite = np.clip(
                            bg_arr + creature.astype(np.float32), 0, 255
                        ).astype(np.uint8)
                        preview_writer.write(composite)

        info['frames_processed'] = frame_num

    finally:
        for w in writers:
            w.__exit__(None, None, None)

    if preview_path:
        info['preview_video'] = str(preview_path)

    return info


# ---------------------------------------------------------------------------
# CLI entry point: python -m wzrd.subtract_video
# ---------------------------------------------------------------------------

def _cli():
    """CLI entry point for ``python -m wzrd.subtract_video``."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Background-subtract a video for projection mapping.',
    )
    parser.add_argument('input_video', help='Path to input video')
    parser.add_argument('background', help='Path to background frame image')
    parser.add_argument('-o', '--output', default=None,
                        help='Output path (default: auto-generated)')
    parser.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD,
                        help=f'Luminance difference threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--ramp', type=int, default=DEFAULT_RAMP,
                        help=f'Soft ramp width above threshold (default: {DEFAULT_RAMP})')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                        help=f'Gamma correction, <1 brightens (default: {DEFAULT_GAMMA})')
    parser.add_argument('--blur-radius', type=float, default=DEFAULT_BLUR_RADIUS,
                        help=f'Gaussian blur radius for mask feathering as fraction of min(H,W) (default: {DEFAULT_BLUR_RADIUS})')
    parser.add_argument('--output-mode', choices=['additive', 'alpha'],
                        default=DEFAULT_OUTPUT_MODE,
                        help=f'Output format (default: {DEFAULT_OUTPUT_MODE})')
    parser.add_argument('--morph-size', type=float, default=DEFAULT_MORPH_SIZE,
                        help=f'Morphological kernel as fraction of min(H,W) (0=off, default: {DEFAULT_MORPH_SIZE})')
    parser.add_argument('--no-color-correction', action='store_true',
                        help='Disable one-time color shift estimation')
    parser.add_argument('--color-correction-threshold', type=float,
                        default=DEFAULT_COLOR_CORRECTION_THRESHOLD,
                        help=f'Max luminance diff for background pixels during color shift estimation (default: {DEFAULT_COLOR_CORRECTION_THRESHOLD})')
    parser.add_argument('--no-subtract-bg', action='store_true',
                        help='Disable background subtraction before masking')
    parser.add_argument('--preview', action='store_true',
                        help='Also save a preview video with composite')
    parser.add_argument('--crf', type=int, default=23,
                        help='FFmpeg CRF quality (default: 23)')
    parser.add_argument('--codec', default='libx264',
                        help='Video codec (default: libx264)')
    args = parser.parse_args()
    
    start_time = time.time()

    def progress(frame_num, total):
        if frame_num % 30 == 0 or frame_num == 1:
            total_str = str(total) if total else '?'
            total_time = time.time() - start_time
            print(f'  Frame {frame_num}/{total_str}, time elapsed: {total_time}', flush=True)

    print(f'Processing: {args.input_video}')
    print(f'Background: {args.background}')

    info = subtract_background_video(
        args.input_video,
        args.background,
        output_path=args.output,
        threshold=args.threshold,
        ramp=args.ramp,
        gamma=args.gamma,
        blur_radius=args.blur_radius,
        output_mode=args.output_mode,
        morph_size=args.morph_size,
        color_correction=not args.no_color_correction,
        color_correction_threshold=args.color_correction_threshold,
        subtract_bg=not args.no_subtract_bg,
        preview=args.preview,
        crf=args.crf,
        codec=args.codec,
        progress_callback=progress,
    )

    print(f'\nDone!')
    print(f'  Output: {info["output_video"]}')
    print(f'  Frames: {info["frames_processed"]}')
    print(f'  Settings: threshold={info["threshold"]}, ramp={info["ramp"]}, '
          f'gamma={info["gamma"]:.2f}')
    if 'preview_video' in info:
        print(f'  Preview: {info["preview_video"]}')


if __name__ == '__main__':
    _cli()
