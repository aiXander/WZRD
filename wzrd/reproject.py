"""
Reproject island videos onto full canvas for projection mapping.

Takes cropped island videos (with their JSON coordinates) and reprojects
them onto a full-resolution canvas, positioned at their original locations.
This enables layer-based compositing in tools like Resolume using additive blending.
"""

import subprocess
import json
from pathlib import Path
from typing import Union, Optional, Tuple, Callable, List, Dict
import numpy as np
from PIL import Image

from .subtract_video import get_video_info, iter_video_frames
from .utils import parse_aspect_ratio, compute_target_dimensions, ASPECT_RATIOS


def load_island_metadata(json_path: Union[str, Path]) -> Dict:
    """
    Load island metadata from JSON file.

    Args:
        json_path: Path to JSON file with island coordinates

    Returns:
        Dict with keys: x, y, width, height, and optionally others
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both single island dict and list format
    if isinstance(data, list):
        if len(data) == 1:
            return data[0]
        raise ValueError(
            f"JSON contains {len(data)} islands. Please specify which island "
            "or provide a single-island JSON file."
        )

    return data


def reproject_video(
    video_path: Union[str, Path],
    island_metadata: Union[str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
    crf: int = 18,
    codec: str = 'libx264',
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Reproject a cropped island video onto a full canvas.

    Places the island video content at its original position on a black
    canvas, creating a video suitable for layer compositing.

    Args:
        video_path: Path to cropped island video
        island_metadata: Path to JSON file or dict with x, y, width, height
        output_path: Output path (auto-generated if None)
        canvas_width: Output canvas width in pixels
        canvas_height: Output canvas height in pixels
        crf: FFmpeg CRF quality (0-51, lower=better)
        codec: Video codec (default: libx264)
        progress_callback: Optional callback(frame_num, total_frames)

    Returns:
        Info dict with processing details
    """
    video_path = Path(video_path)

    # Load metadata
    if isinstance(island_metadata, (str, Path)):
        metadata = load_island_metadata(island_metadata)
    else:
        metadata = island_metadata

    island_x = metadata['x']
    island_y = metadata['y']
    island_w = metadata['width']
    island_h = metadata['height']

    # Get video info
    vid_width, vid_height, fps, frame_count = get_video_info(video_path)

    # Generate output path
    if output_path is None:
        output_path = video_path.with_name(
            f"{video_path.stem}_reprojected_{canvas_width}x{canvas_height}.mp4"
        )
    output_path = Path(output_path)

    info = {
        'input_video': str(video_path),
        'output_video': str(output_path),
        'video_size': (vid_width, vid_height),
        'island_position': (island_x, island_y),
        'island_size': (island_w, island_h),
        'canvas_size': (canvas_width, canvas_height),
        'fps': fps,
        'frame_count': frame_count,
    }

    # Check if island fits on canvas
    if island_x + island_w > canvas_width or island_y + island_h > canvas_height:
        info['warning'] = (
            f"Island extends beyond canvas: island ends at "
            f"({island_x + island_w}, {island_y + island_h}) but canvas is "
            f"{canvas_width}x{canvas_height}. Island will be clipped."
        )

    # Setup FFmpeg output pipe
    output_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{canvas_width}x{canvas_height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', codec,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-v', 'error',
        str(output_path)
    ]

    output_process = subprocess.Popen(
        output_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Create black canvas template
    canvas_template = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    frame_num = 0
    try:
        for fn, frame_arr in iter_video_frames(video_path):
            frame_num = fn

            if progress_callback:
                progress_callback(frame_num, frame_count or 0)

            # Create canvas with island placed at correct position
            canvas = canvas_template.copy()

            # Calculate placement bounds (handle edge clipping)
            src_x_start = 0
            src_y_start = 0
            src_x_end = frame_arr.shape[1]
            src_y_end = frame_arr.shape[0]

            dst_x_start = island_x
            dst_y_start = island_y
            dst_x_end = island_x + frame_arr.shape[1]
            dst_y_end = island_y + frame_arr.shape[0]

            # Clip to canvas bounds
            if dst_x_start < 0:
                src_x_start = -dst_x_start
                dst_x_start = 0
            if dst_y_start < 0:
                src_y_start = -dst_y_start
                dst_y_start = 0
            if dst_x_end > canvas_width:
                src_x_end -= (dst_x_end - canvas_width)
                dst_x_end = canvas_width
            if dst_y_end > canvas_height:
                src_y_end -= (dst_y_end - canvas_height)
                dst_y_end = canvas_height

            # Place frame on canvas
            if dst_x_end > dst_x_start and dst_y_end > dst_y_start:
                canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    frame_arr[src_y_start:src_y_end, src_x_start:src_x_end]

            # Write to output
            output_process.stdin.write(canvas.tobytes())

        info['frames_processed'] = frame_num

    finally:
        output_process.stdin.close()
        output_process.wait()

    return info


def reproject_videos_batch(
    video_json_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
    output_dir: Optional[Union[str, Path]] = None,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
    crf: int = 18,
    codec: str = 'libx264',
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
) -> List[Dict]:
    """
    Reproject multiple island videos onto a common canvas.

    All output videos will have identical resolution, suitable for
    layer-based compositing with additive blending.

    Args:
        video_json_pairs: List of (video_path, json_path) tuples
        output_dir: Directory for output videos (default: same as input)
        canvas_width: Output canvas width
        canvas_height: Output canvas height
        crf: FFmpeg CRF quality
        codec: Video codec
        progress_callback: Optional callback(video_num, total_videos, frame_num, total_frames)

    Returns:
        List of info dicts, one per video
    """
    results = []
    total_videos = len(video_json_pairs)

    for video_idx, (video_path, json_path) in enumerate(video_json_pairs, 1):
        video_path = Path(video_path)

        # Determine output path
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = output_dir_path / f"{video_path.stem}_reprojected.mp4"
        else:
            output_path = None

        # Create per-video progress wrapper
        def video_progress(frame_num, total_frames):
            if progress_callback:
                progress_callback(video_idx, total_videos, frame_num, total_frames)

        info = reproject_video(
            video_path=video_path,
            island_metadata=json_path,
            output_path=output_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            crf=crf,
            codec=codec,
            progress_callback=video_progress,
        )

        results.append(info)

    return results


def reproject_from_islands_dir(
    islands_dir: Union[str, Path],
    video_pattern: str = "island_*.mp4",
    output_dir: Optional[Union[str, Path]] = None,
    canvas_width: int = 1920,
    canvas_height: int = 1080,
    crf: int = 18,
    codec: str = 'libx264',
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
) -> List[Dict]:
    """
    Reproject all island videos from a directory using islands.json metadata.

    Expects directory structure from save_islands():
    - islands_dir/
      - islands.json (contains all island metadata)
      - island_1.mp4 (or matching video_pattern)
      - island_2.mp4
      - ...

    Args:
        islands_dir: Directory containing island videos and islands.json
        video_pattern: Glob pattern for video files (default: island_*.mp4)
        output_dir: Output directory (default: islands_dir/reprojected)
        canvas_width: Output canvas width
        canvas_height: Output canvas height
        crf: FFmpeg CRF quality
        codec: Video codec
        progress_callback: Optional callback(video_num, total_videos, frame_num, total_frames)

    Returns:
        List of info dicts
    """
    islands_dir = Path(islands_dir)

    # Load master islands.json
    islands_json_path = islands_dir / 'islands.json'
    if not islands_json_path.exists():
        raise FileNotFoundError(
            f"islands.json not found in {islands_dir}. "
            "Use reproject_videos_batch() with explicit video/json pairs instead."
        )

    with open(islands_json_path, 'r') as f:
        islands_data = json.load(f)

    # Create lookup by id and filename
    islands_by_id = {isl['id']: isl for isl in islands_data}
    islands_by_filename = {}
    for isl in islands_data:
        if 'filename' in isl:
            # Strip extension to match video files
            base_name = Path(isl['filename']).stem
            islands_by_filename[base_name] = isl

    # Find all video files
    video_files = sorted(islands_dir.glob(video_pattern))
    if not video_files:
        raise FileNotFoundError(
            f"No videos matching '{video_pattern}' found in {islands_dir}"
        )

    # Match videos to metadata
    video_json_pairs = []
    for video_path in video_files:
        video_stem = video_path.stem

        # Try to find matching metadata
        metadata = None

        # First try direct filename match
        if video_stem in islands_by_filename:
            metadata = islands_by_filename[video_stem]
        else:
            # Try to extract island ID from filename (e.g., "island_1" -> 1)
            for part in video_stem.split('_'):
                try:
                    island_id = int(part)
                    if island_id in islands_by_id:
                        metadata = islands_by_id[island_id]
                        break
                except ValueError:
                    continue

        if metadata is None:
            raise ValueError(
                f"Could not find metadata for video: {video_path}. "
                f"Available islands: {list(islands_by_id.keys())}"
            )

        video_json_pairs.append((video_path, metadata))

    # Set default output directory
    if output_dir is None:
        output_dir = islands_dir / 'reprojected'

    # Process all videos
    results = []
    total_videos = len(video_json_pairs)

    for video_idx, (video_path, metadata) in enumerate(video_json_pairs, 1):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"{video_path.stem}_reprojected.mp4"

        def video_progress(frame_num, total_frames):
            if progress_callback:
                progress_callback(video_idx, total_videos, frame_num, total_frames)

        info = reproject_video(
            video_path=video_path,
            island_metadata=metadata,
            output_path=output_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            crf=crf,
            codec=codec,
            progress_callback=video_progress,
        )

        results.append(info)

    return results


def reproject_video_with_aspect(
    video_path: Union[str, Path],
    island_metadata: Union[str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    target_aspect: Union[str, float] = "16:9",
    base_resolution: int = 1920,
    crf: int = 18,
    codec: str = 'libx264',
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    """
    Reproject island video using aspect ratio and base resolution.

    Convenience wrapper that computes canvas dimensions from aspect ratio.

    Args:
        video_path: Path to cropped island video
        island_metadata: Path to JSON or dict with coordinates
        output_path: Output path (auto-generated if None)
        target_aspect: Target aspect ratio (e.g., "16:9" or 1.7778)
        base_resolution: Max dimension of output canvas
        crf: FFmpeg CRF quality
        codec: Video codec
        progress_callback: Optional callback(frame_num, total_frames)

    Returns:
        Info dict with processing details
    """
    # Parse aspect ratio
    if isinstance(target_aspect, str):
        aspect = parse_aspect_ratio(target_aspect)
    else:
        aspect = target_aspect

    # Compute canvas dimensions
    canvas_width, canvas_height = compute_target_dimensions(aspect, base_resolution)

    return reproject_video(
        video_path=video_path,
        island_metadata=island_metadata,
        output_path=output_path,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        crf=crf,
        codec=codec,
        progress_callback=progress_callback,
    )


def _cli():
    """CLI entry point for ``python -m wzrd.reproject``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Reproject island videos onto full canvas for layer compositing'
    )
    parser.add_argument('input', nargs='+',
                        help='Video file(s) with JSON, or directory containing islands')
    parser.add_argument('-o', '--output', help='Output path/directory')
    parser.add_argument('--width', type=int, default=1920,
                        help='Canvas width in pixels. Default: 1920')
    parser.add_argument('--height', type=int, default=1080,
                        help='Canvas height in pixels. Default: 1080')
    parser.add_argument('--aspect', type=str, default=None,
                        help='Target aspect ratio (e.g., 16:9). Overrides width/height.')
    parser.add_argument('--base-resolution', type=int, default=1920,
                        help='Base resolution when using --aspect. Default: 1920')
    parser.add_argument('--crf', type=int, default=18,
                        help='FFmpeg CRF quality. Default: 18')
    parser.add_argument('--codec', type=str, default='libx264',
                        help='Video codec. Default: libx264')
    args = parser.parse_args()

    # Compute canvas size from aspect ratio if provided
    if args.aspect:
        aspect = parse_aspect_ratio(args.aspect)
        args.width, args.height = compute_target_dimensions(aspect, args.base_resolution)
        print(f"Using aspect {args.aspect} -> {args.width}x{args.height}")

    def progress(video_num, total_videos, frame_num, total_frames):
        total_str = str(total_frames) if total_frames else '?'
        if frame_num % 30 == 0 or frame_num == 1:
            print(f"  [{video_num}/{total_videos}] Frame {frame_num}/{total_str}")

    def single_progress(frame_num, total_frames):
        total_str = str(total_frames) if total_frames else '?'
        if frame_num % 30 == 0 or frame_num == 1:
            print(f"  Frame {frame_num}/{total_str}")

    # Check if input is a directory with islands
    input_path = Path(args.input[0])

    if input_path.is_dir():
        print(f"Processing islands directory: {input_path}")
        results = reproject_from_islands_dir(
            islands_dir=input_path,
            output_dir=args.output,
            canvas_width=args.width,
            canvas_height=args.height,
            crf=args.crf,
            codec=args.codec,
            progress_callback=progress,
        )
        print(f"\nComplete! Processed {len(results)} videos")
        for info in results:
            print(f"  {info['output_video']}")

    elif len(args.input) == 1:
        video_path = input_path
        json_path = video_path.with_suffix('.json')
        if not json_path.exists():
            json_path = video_path.parent / 'islands.json'

        if not json_path.exists():
            print(f"Error: Could not find JSON metadata for {video_path}")
            print(f"  Tried: {video_path.with_suffix('.json')}")
            print(f"  Tried: {video_path.parent / 'islands.json'}")
            sys.exit(1)

        print(f"Processing: {video_path}")
        print(f"  Metadata: {json_path}")
        info = reproject_video(
            video_path=video_path,
            island_metadata=json_path,
            output_path=args.output,
            canvas_width=args.width,
            canvas_height=args.height,
            crf=args.crf,
            codec=args.codec,
            progress_callback=single_progress,
        )
        print(f"\nComplete!")
        print(f"  Output: {info['output_video']}")
        print(f"  Canvas: {info['canvas_size'][0]}x{info['canvas_size'][1]}")
        print(f"  Island position: {info['island_position']}")

    else:
        if len(args.input) % 2 != 0:
            print("Error: When providing multiple files, provide video/json pairs")
            print("  Example: python -m wzrd.reproject video1.mp4 video1.json video2.mp4 video2.json")
            sys.exit(1)

        pairs = []
        for i in range(0, len(args.input), 2):
            video_path = Path(args.input[i])
            json_path = Path(args.input[i + 1])
            pairs.append((video_path, json_path))

        print(f"Processing {len(pairs)} video/json pairs")
        results = reproject_videos_batch(
            video_json_pairs=pairs,
            output_dir=args.output,
            canvas_width=args.width,
            canvas_height=args.height,
            crf=args.crf,
            codec=args.codec,
            progress_callback=progress,
        )
        print(f"\nComplete! Processed {len(results)} videos")
        for info in results:
            print(f"  {info['output_video']}")


if __name__ == '__main__':
    _cli()
