"""
Command-line interface for WZRD tools.

These are optional CLI entry points. The core functionality
is available via the Python API in the other modules.
"""

import argparse
import sys
from pathlib import Path


def darken_cli():
    """CLI for darkening images."""
    from .darken import darken_image_file
    from .config import get_darken_config, get_resolution_config

    darken_config = get_darken_config()
    resolution_config = get_resolution_config()

    parser = argparse.ArgumentParser(description='Darken an image for projection mapping')
    parser.add_argument('input_image', help='Path to the input image')
    parser.add_argument('-o', '--output', help='Output path (default: auto-generated)')
    parser.add_argument('--gamma', type=float, default=None,
                        help=f'Dynamic range compression. Default: {darken_config["gamma"]}')
    parser.add_argument('--max-brightness', type=float, default=None,
                        help=f'Maximum output intensity (0.0-1.0). Default: {darken_config["max_brightness"]}')
    parser.add_argument('--aspect', type=str, default=None,
                        help=f'Target aspect ratio (e.g., 16:9). Default: {resolution_config["default_aspect"]}')
    parser.add_argument('--base-resolution', type=int, default=None,
                        help=f'Base resolution. Default: {resolution_config["base_resolution"]}')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip aspect ratio and resolution normalization')
    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        gamma = args.gamma or darken_config['gamma']
        input_path = Path(args.input_image)
        args.output = str(input_path.with_name(f"{input_path.stem}_darkened_{gamma:.2f}.jpg"))

    img, info = darken_image_file(
        args.input_image,
        output_path=args.output,
        gamma=args.gamma,
        max_brightness=args.max_brightness,
        target_aspect=args.aspect,
        base_resolution=args.base_resolution,
        normalize=not args.no_normalize,
    )

    print(f"Saved: {info.get('output_path', args.output)}")
    print(f"  Size: {info['final_size'][0]}x{info['final_size'][1]}")
    print(f"  Gamma: {info['gamma']}, Max brightness: {info['max_brightness']}")


def subtract_cli():
    """CLI for background subtraction on images."""
    from .subtract import subtract_background_file
    from .config import get_subtract_config

    config = get_subtract_config()

    parser = argparse.ArgumentParser(description='Subtract background from generated keyframe')
    parser.add_argument('generated_frame', help='Path to the AI-generated keyframe (with creature)')
    parser.add_argument('background_frame', help='Path to the original darkened background frame')
    parser.add_argument('-o', '--output', help='Output path (default: auto-generated)')
    parser.add_argument('--threshold', type=int, default=None,
                        help=f'Difference threshold (0-255). Default: {config["threshold"]}')
    parser.add_argument('--boost', type=float, default=None,
                        help=f'Brightness boost multiplier. Default: {config["boost"]}')
    parser.add_argument('--feather', type=int, default=None,
                        help=f'Edge feather radius. Default: {config["feather_radius"]}')
    parser.add_argument('--mode', type=str, choices=['rgb', 'lab', 'luminance'], default=None,
                        help=f'Difference calculation mode. Default: {config["diff_mode"]}')
    parser.add_argument('--output-mode', type=str, choices=['additive', 'alpha'], default=None,
                        help=f'Output format. Default: {config["output_mode"]}')
    parser.add_argument('--preview', action='store_true',
                        help='Also save a preview composite')
    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        threshold = args.threshold or config['threshold']
        boost = args.boost or config['boost']
        feather = args.feather or config['feather_radius']
        mode = args.mode or config['diff_mode']
        output_mode = args.output_mode or config['output_mode']

        input_path = Path(args.generated_frame)
        params_str = f"t{threshold}_b{boost:.1f}_f{feather}_{mode}"
        suffix = '_alpha.png' if output_mode == 'alpha' else '.png'
        args.output = str(input_path.with_name(f"{input_path.stem}_creature_{params_str}{suffix}"))

    img, info = subtract_background_file(
        args.generated_frame,
        args.background_frame,
        output_path=args.output,
        threshold=args.threshold,
        boost=args.boost,
        feather_radius=args.feather,
        diff_mode=args.mode,
        output_mode=args.output_mode,
        preview=args.preview,
    )

    print(f"Saved: {info.get('output_path', args.output)}")
    print(f"  Mask coverage: {info['mask_coverage']:.1f}%")
    print(f"  Max brightness: {info['creature_max_brightness']}")
    if args.preview and 'preview_path' in info:
        print(f"  Preview: {info['preview_path']}")


def subtract_video_cli():
    """CLI for background subtraction on videos."""
    from .video import subtract_background_video
    from .config import get_subtract_config

    config = get_subtract_config()

    parser = argparse.ArgumentParser(description='Subtract background from video')
    parser.add_argument('input_video', help='Path to the input video')
    parser.add_argument('background_frame', help='Path to the background frame image')
    parser.add_argument('-o', '--output', help='Output path (default: auto-generated)')
    parser.add_argument('--threshold', type=int, default=None,
                        help=f'Difference threshold. Default: {config["threshold"]}')
    parser.add_argument('--boost', type=float, default=None,
                        help=f'Brightness boost. Default: {config["boost"]}')
    parser.add_argument('--feather', type=int, default=None,
                        help=f'Edge feather radius. Default: {config["feather_radius"]}')
    parser.add_argument('--mode', type=str, choices=['rgb', 'lab', 'luminance'], default=None,
                        help=f'Difference mode. Default: {config["diff_mode"]}')
    parser.add_argument('--output-mode', type=str, choices=['additive', 'alpha'], default=None,
                        help=f'Output format. Default: {config["output_mode"]}')
    parser.add_argument('--preview', action='store_true',
                        help='Also save a preview video')
    parser.add_argument('--crf', type=int, default=18,
                        help='FFmpeg CRF quality. Default: 18')
    parser.add_argument('--codec', type=str, default='libx264',
                        help='Video codec. Default: libx264')
    args = parser.parse_args()

    def progress(frame_num, total):
        if frame_num % 30 == 0 or frame_num == 1:
            total_str = str(total) if total else '?'
            print(f"  Processing frame {frame_num}/{total_str}")

    info = subtract_background_video(
        args.input_video,
        args.background_frame,
        output_path=args.output,
        threshold=args.threshold,
        boost=args.boost,
        feather_radius=args.feather,
        diff_mode=args.mode,
        output_mode=args.output_mode,
        preview=args.preview,
        crf=args.crf,
        codec=args.codec,
        progress_callback=progress,
    )

    print(f"\nComplete!")
    print(f"  Output: {info['output_video']}")
    print(f"  Frames: {info['frames_processed']}")
    if 'preview_video' in info:
        print(f"  Preview: {info['preview_video']}")


def crop_islands_cli():
    """CLI for extracting islands from segmentation maps."""
    from .islands import crop_islands_file

    parser = argparse.ArgumentParser(description='Extract islands from binary segmentation map')
    parser.add_argument('input_image', help='Path to the binary segmentation map')
    parser.add_argument('-o', '--output', help='Output directory (default: auto-generated)')
    parser.add_argument('--min-area', type=int, default=0,
                        help='Minimum area in pixels. Default: 0')
    parser.add_argument('--connectivity', type=int, choices=[4, 8], default=8,
                        help='Pixel connectivity. Default: 8')
    args = parser.parse_args()

    islands, output_dir = crop_islands_file(
        args.input_image,
        output_dir=args.output,
        min_area=args.min_area,
        connectivity=args.connectivity,
    )

    print(f"Found {len(islands)} islands")
    print(f"Output directory: {output_dir}")
    for island in islands:
        print(f"  Island {island['id']}: {island['width']}x{island['height']} @ ({island['x']},{island['y']})")


if __name__ == '__main__':
    print("Use the CLI entry points: wzrd-darken, wzrd-subtract, wzrd-subtract-video, wzrd-crop-islands")
