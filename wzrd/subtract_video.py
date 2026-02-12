"""
Video processing for background subtraction.

Functions to process video files frame-by-frame, applying background
subtraction to extract creatures/elements across all frames.
"""

from PIL import Image
import numpy as np
import subprocess
from pathlib import Path
from typing import Union, Optional, Tuple, Literal, Callable, Iterator, List
import cv2
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from .utils import (
    align_to_reference, get_aspect_ratio, aspect_ratios_match,
    correct_color_shift, compute_difference_mask, extract_creature,
)
# Module-level defaults (previously loaded from config.yaml)
DEFAULT_THRESHOLD = 10
DEFAULT_RAMP = 20
DEFAULT_GAMMA = 0.85
DEFAULT_FEATHER_RADIUS = 4
DEFAULT_DIFF_MODE = 'lab'
DEFAULT_OUTPUT_MODE = 'additive'
DEFAULT_MIN_ALPHA = 0.0
DEFAULT_MORPH_SIZE = 5
DEFAULT_GUIDED_FILTER_EPS = 0.02
DEFAULT_COLOR_CORRECTION_PERCENTILE = 50
DEFAULT_TEMPORAL_SMOOTHING = 0.4
DEFAULT_ASPECT_TOLERANCE = 0.02


DiffMode = Literal['rgb', 'lab', 'luminance']
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


def process_frame(
    gen_arr: np.ndarray,
    bg_arr: np.ndarray,
    threshold: int = 10,
    ramp: int = 20,
    gamma: float = 0.85,
    feather_radius: int = 4,
    diff_mode: DiffMode = 'lab',
    min_alpha: float = 0.0,
    morph_size: int = 5,
    guided_filter_eps: float = 0.02,
    color_correction_percentile: float = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single frame: color-correct, compute mask, extract creature.

    Args:
        gen_arr:    Generated frame float32 (HWC, 0-255).
        bg_arr:     Background frame float32 (HWC, 0-255).
        threshold:  Low cutoff for difference mask.
        ramp:       Soft transition width above threshold.
        gamma:      Gamma correction (< 1.0 brightens).
        feather_radius: Guided-filter radius.
        diff_mode:  ``'rgb'``, ``'lab'``, or ``'luminance'``.
        min_alpha:  Minimum mask value.
        morph_size: Morphological kernel size (0 = disable).
        guided_filter_eps: Guided-filter regularization.
        color_correction_percentile: Color correction strength (0 = disable).

    Returns:
        Tuple of (creature uint8 array, mask float32 array).
    """
    if color_correction_percentile > 0:
        gen_arr, _ = correct_color_shift(
            gen_arr, bg_arr, color_correction_percentile
        )

    mask = compute_difference_mask(
        gen_arr, bg_arr,
        threshold=threshold,
        ramp=ramp,
        feather_radius=feather_radius,
        diff_mode=diff_mode,
        min_alpha=min_alpha,
        morph_size=morph_size,
        guided_filter_eps=guided_filter_eps,
    )

    creature = extract_creature(gen_arr, bg_arr, mask, gamma=gamma)
    return creature, mask


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
                frame_pil = Image.fromarray(frame_arr)
                frame_pil = frame_pil.resize(
                    target_size, Image.Resampling.LANCZOS
                )
                frame_arr = np.array(frame_pil)

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
    feather_radius: int = DEFAULT_FEATHER_RADIUS,
    diff_mode: DiffMode = DEFAULT_DIFF_MODE,
    output_mode: OutputMode = DEFAULT_OUTPUT_MODE,
    morph_size: int = DEFAULT_MORPH_SIZE,
    guided_filter_eps: float = DEFAULT_GUIDED_FILTER_EPS,
    color_correction_percentile: float = DEFAULT_COLOR_CORRECTION_PERCENTILE,
    temporal_smoothing: float = DEFAULT_TEMPORAL_SMOOTHING,
    preview: bool = False,
    crf: int = 18,
    codec: str = 'libx264',
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Process a video file with background subtraction.

    Uses streaming FFmpeg pipes for memory efficiency.  Expensive per-frame
    work (color correction, LAB conversion, mask computation) runs in
    parallel threads; temporal smoothing and creature extraction are applied
    sequentially to maintain frame ordering.

    Args:
        video_path:  Path to input video.
        background_path: Path to background frame image.
        output_path: Path for output video (auto-generated if None).
        threshold:   Difference threshold (0-255).
        ramp:        Soft ramp width.
        gamma:       Gamma correction (< 1.0 brightens).
        feather_radius: Edge feather radius.
        diff_mode:   Difference mode ('rgb', 'lab', 'luminance').
        output_mode: ``'additive'`` or ``'alpha'``.
        morph_size:  Morphological kernel size (0 = disable).
        guided_filter_eps: Guided-filter regularization.
        color_correction_percentile: Color correction (0 = disable).
        temporal_smoothing: EMA smoothing strength 0-1 where higher = more
            smoothing. 0 disables.
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
    aspect_tolerance = DEFAULT_ASPECT_TOLERANCE

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
    needs_alignment = (width, height) != bg_size

    # Setup output paths
    if output_path is None:
        params_str = (
            f"t{threshold}_g{gamma:.2f}_f{feather_radius}_{diff_mode}"
        )
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
        'output_size': bg_size,
        'fps': fps,
        'frame_count': frame_count,
        'threshold': threshold,
        'ramp': ramp,
        'gamma': gamma,
        'feather_radius': feather_radius,
        'diff_mode': diff_mode,
        'output_mode': output_mode,
        'morph_size': morph_size,
        'temporal_smoothing': temporal_smoothing,
        'color_correction_percentile': color_correction_percentile,
        'needs_alignment': needs_alignment,
    }

    # ---- FFmpeg output setup ------------------------------------------------
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
        str(output_path),
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
            str(preview_path),
        ]

    # Start output processes
    output_process = subprocess.Popen(
        output_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    preview_process = (
        subprocess.Popen(
            preview_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if preview_cmd else None
    )

    # ---- Parallel batch processing ------------------------------------------
    num_workers = min(cpu_count(), 8)
    batch_size = num_workers * 2

    def _compute_frame_data(args):
        """Thread worker: color-correct and compute raw mask."""
        fn, frame_arr = args
        gen_arr = frame_arr.astype(np.float32)

        # Color correction
        if color_correction_percentile > 0:
            gen_arr, _ = correct_color_shift(
                gen_arr, bg_arr, color_correction_percentile
            )

        # Compute raw mask (temporal smoothing applied sequentially later)
        mask = compute_difference_mask(
            gen_arr, bg_arr,
            threshold=threshold,
            ramp=ramp,
            feather_radius=feather_radius,
            diff_mode=diff_mode,
            min_alpha=0.0,          # applied after temporal smoothing
            morph_size=morph_size,
            guided_filter_eps=guided_filter_eps,
        )

        return fn, gen_arr, mask

    prev_mask = None
    frame_num = 0

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for batch in iter_video_frames_batched(
                video_path, batch_size=batch_size
            ):
                # Align frames (sequential — uses PIL)
                aligned_batch = []
                for fn, frame_arr in batch:
                    if needs_alignment:
                        frame_pil = Image.fromarray(frame_arr)
                        frame_pil, _ = align_to_reference(
                            frame_pil, background, aspect_tolerance
                        )
                        frame_arr = np.array(frame_pil)
                    aligned_batch.append((fn, frame_arr))

                # Parallel: color correction + mask computation
                results = list(
                    executor.map(_compute_frame_data, aligned_batch)
                )
                results.sort(key=lambda x: x[0])

                # Sequential: temporal smoothing → creature extraction → write
                for fn, gen_arr, raw_mask in results:
                    frame_num = fn

                    # Temporal EMA smoothing
                    if temporal_smoothing > 0 and prev_mask is not None:
                        mask = (
                            (1 - temporal_smoothing) * raw_mask
                            + temporal_smoothing * prev_mask
                        )
                    else:
                        mask = raw_mask
                    prev_mask = mask.copy()

                    # Apply min_alpha after temporal smoothing
                    if min_alpha > 0:
                        mask = np.clip(mask, min_alpha, 1.0)

                    # Extract creature
                    creature = extract_creature(
                        gen_arr, bg_arr, mask, gamma=gamma
                    )

                    if progress_callback:
                        progress_callback(frame_num, frame_count or 0)

                    # Write to output pipe
                    if output_mode == 'alpha':
                        alpha_channel = (mask * 255).astype(np.uint8)
                        rgba = np.dstack([creature, alpha_channel])
                        output_process.stdin.write(rgba.tobytes())
                    else:
                        output_process.stdin.write(creature.tobytes())

                    # Write preview
                    if preview_process:
                        composite = np.clip(
                            bg_arr + creature.astype(np.float32), 0, 255
                        ).astype(np.uint8)
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
                        help=f'Low cutoff for difference mask (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--ramp', type=int, default=DEFAULT_RAMP,
                        help=f'Soft ramp width above threshold (default: {DEFAULT_RAMP})')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                        help=f'Gamma correction, <1 brightens (default: {DEFAULT_GAMMA})')
    parser.add_argument('--feather', type=int, default=DEFAULT_FEATHER_RADIUS,
                        help=f'Guided-filter feather radius (default: {DEFAULT_FEATHER_RADIUS})')
    parser.add_argument('--mode', choices=['rgb', 'lab', 'luminance'],
                        default=DEFAULT_DIFF_MODE,
                        help=f'Difference mode (default: {DEFAULT_DIFF_MODE})')
    parser.add_argument('--output-mode', choices=['additive', 'alpha'],
                        default=DEFAULT_OUTPUT_MODE,
                        help=f'Output format (default: {DEFAULT_OUTPUT_MODE})')
    parser.add_argument('--morph-size', type=int, default=DEFAULT_MORPH_SIZE,
                        help=f'Morphological cleanup kernel (0=off, default: {DEFAULT_MORPH_SIZE})')
    parser.add_argument('--color-correction', type=float,
                        default=DEFAULT_COLOR_CORRECTION_PERCENTILE,
                        help=f'Color correction percentile (0=off, default: {DEFAULT_COLOR_CORRECTION_PERCENTILE})')
    parser.add_argument('--temporal-smoothing', type=float,
                        default=DEFAULT_TEMPORAL_SMOOTHING,
                        help=f'Temporal EMA smoothing 0-1, higher=smoother (0=off, default: {DEFAULT_TEMPORAL_SMOOTHING})')
    parser.add_argument('--guided-filter-eps', type=float,
                        default=DEFAULT_GUIDED_FILTER_EPS,
                        help=f'Guided-filter epsilon (default: {DEFAULT_GUIDED_FILTER_EPS})')
    parser.add_argument('--preview', action='store_true',
                        help='Also save a preview video with composite')
    parser.add_argument('--crf', type=int, default=18,
                        help='FFmpeg CRF quality (default: 18)')
    parser.add_argument('--codec', default='libx264',
                        help='Video codec (default: libx264)')
    args = parser.parse_args()

    def progress(frame_num, total):
        if frame_num % 30 == 0 or frame_num == 1:
            total_str = str(total) if total else '?'
            print(f'  Frame {frame_num}/{total_str}', flush=True)

    print(f'Processing: {args.input_video}')
    print(f'Background: {args.background}')

    info = subtract_background_video(
        args.input_video,
        args.background,
        output_path=args.output,
        threshold=args.threshold,
        ramp=args.ramp,
        gamma=args.gamma,
        feather_radius=args.feather,
        diff_mode=args.mode,
        output_mode=args.output_mode,
        morph_size=args.morph_size,
        guided_filter_eps=args.guided_filter_eps,
        color_correction_percentile=args.color_correction,
        temporal_smoothing=args.temporal_smoothing,
        preview=args.preview,
        crf=args.crf,
        codec=args.codec,
        progress_callback=progress,
    )

    print(f'\nDone!')
    print(f'  Output: {info["output_video"]}')
    print(f'  Frames: {info["frames_processed"]}')
    print(f'  Settings: threshold={info["threshold"]}, ramp={info["ramp"]}, '
          f'gamma={info["gamma"]:.2f}, mode={info["diff_mode"]}, '
          f'temporal_smoothing={info["temporal_smoothing"]}')
    if 'preview_video' in info:
        print(f'  Preview: {info["preview_video"]}')


if __name__ == '__main__':
    _cli()
