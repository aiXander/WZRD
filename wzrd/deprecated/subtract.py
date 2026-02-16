"""
Background subtraction for creature extraction.

Functions to isolate bright elements (creatures) from a dark background
by computing the difference between generated frames and the original background.
"""

from PIL import Image, ImageFilter
import numpy as np
from typing import Union, Optional, Tuple, Literal
from pathlib import Path

from .utils import align_to_reference, get_aspect_ratio, aspect_ratios_match

# Module-level defaults (inline, no config dependency)
DEFAULT_THRESHOLD = 10
DEFAULT_BOOST = 1.1
DEFAULT_FEATHER_RADIUS = 4
DEFAULT_DIFF_MODE = 'luminance'
DEFAULT_OUTPUT_MODE = 'additive'
DEFAULT_MIN_ALPHA = 0.0
DEFAULT_ASPECT_TOLERANCE = 0.02


DiffMode = Literal['rgb', 'lab', 'luminance']
OutputMode = Literal['additive', 'alpha']


def compute_difference_mask(
    generated: np.ndarray,
    background: np.ndarray,
    threshold: int = 10,
    feather_radius: int = 4,
    diff_mode: DiffMode = 'luminance',
    min_alpha: float = 0.0,
) -> np.ndarray:
    """
    Compute a difference mask between generated and background frames.

    Args:
        generated: Generated frame as float32 array (HWC, 0-255)
        background: Background frame as float32 array (HWC, 0-255)
        threshold: Difference threshold (0-255). Pixels below this are background.
        feather_radius: Edge feather radius in pixels (0 = no feathering)
        diff_mode: 'rgb', 'lab', or 'luminance'
        min_alpha: Minimum mask value (0.0-1.0)

    Returns:
        Mask as float32 array (HW, 0.0-1.0)
    """
    if diff_mode == 'rgb':
        # Per-channel difference, take max across channels
        diff = np.abs(generated - background)
        diff_magnitude = np.max(diff, axis=2)

    elif diff_mode == 'lab':
        # Convert to LAB for perceptually uniform difference
        try:
            from skimage import color
        except ImportError:
            raise ImportError(
                "scikit-image is required for LAB mode. "
                "Install with: pip install scikit-image"
            )
        gen_lab = color.rgb2lab(generated / 255.0)
        bg_lab = color.rgb2lab(background / 255.0)
        # Euclidean distance in LAB space
        diff_magnitude = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))
        # Normalize to 0-255 range (LAB diff can be large)
        diff_magnitude = np.clip(diff_magnitude * 2.55, 0, 255)

    elif diff_mode == 'luminance':
        # Convert to grayscale and compare luminance
        gen_lum = 0.299 * generated[:, :, 0] + 0.587 * generated[:, :, 1] + 0.114 * generated[:, :, 2]
        bg_lum = 0.299 * background[:, :, 0] + 0.587 * background[:, :, 1] + 0.114 * background[:, :, 2]
        diff_magnitude = np.abs(gen_lum - bg_lum)
    else:
        raise ValueError(f"Unknown diff_mode: {diff_mode}")

    # Binary mask based on threshold
    mask = (diff_magnitude > threshold).astype(np.float32)

    # Apply feathering if requested
    if feather_radius > 0:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        mask = np.array(mask_img, dtype=np.float32) / 255.0

    # Apply minimum alpha
    mask = np.clip(mask, min_alpha, 1.0)

    return mask


def extract_creature(
    generated: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    boost: float = 1.1,
) -> np.ndarray:
    """
    Extract the creature (bright additive elements) using the mask.

    Args:
        generated: Generated frame as float32 array (HWC, 0-255)
        background: Background frame as float32 array (HWC, 0-255)
        mask: Difference mask as float32 array (HW, 0.0-1.0)
        boost: Brightness boost multiplier

    Returns:
        Extracted creature as uint8 array (HWC, 0-255)
    """
    # Compute the actual difference (what was added)
    creature_diff = generated - background

    # Clip to positive values only (we want additive elements)
    creature_diff = np.clip(creature_diff, 0, 255)

    # Apply boost
    creature_diff = creature_diff * boost

    # Apply mask to isolate creature
    creature = creature_diff * mask[:, :, np.newaxis]

    # Clip final result
    return np.clip(creature, 0, 255).astype(np.uint8)


def subtract_background(
    generated: Union[Image.Image, np.ndarray],
    background: Union[Image.Image, np.ndarray],
    threshold: int = 10,
    boost: float = 1.1,
    feather_radius: int = 4,
    diff_mode: DiffMode = 'luminance',
    min_alpha: float = 0.0,
    output_mode: OutputMode = 'additive',
    align: bool = True,
    aspect_tolerance: float = 0.02,
) -> Tuple[Union[np.ndarray, Image.Image], np.ndarray, dict]:
    """
    Subtract background from generated frame to isolate creature.

    Args:
        generated: Generated frame (PIL Image or numpy array)
        background: Background frame (PIL Image or numpy array)
        threshold: Difference threshold (0-255)
        boost: Brightness boost multiplier
        feather_radius: Edge feather radius in pixels
        diff_mode: 'rgb', 'lab', or 'luminance'
        min_alpha: Minimum alpha value
        output_mode: 'additive' (RGB on black) or 'alpha' (RGBA with transparency)
        align: Whether to align generated to match background dimensions
        aspect_tolerance: Tolerance for aspect ratio matching

    Returns:
        Tuple of (creature_image, mask, info_dict)
        - creature_image: Extracted creature as numpy array
        - mask: The computed mask (HW, 0.0-1.0)
        - info_dict: Processing information
    """
    info = {
        'threshold': threshold,
        'boost': boost,
        'feather_radius': feather_radius,
        'diff_mode': diff_mode,
        'output_mode': output_mode,
        'aligned': False,
        'cropped': False,
    }

    # Convert to PIL if needed
    if isinstance(background, np.ndarray):
        background = Image.fromarray(background.astype(np.uint8))
    background = background.convert('RGB')

    if isinstance(generated, np.ndarray):
        generated = Image.fromarray(generated.astype(np.uint8))
    generated = generated.convert('RGB')

    # Align generated to background if needed
    if align and generated.size != background.size:
        generated, align_info = align_to_reference(generated, background, aspect_tolerance)
        info['aligned'] = align_info['resized']
        info['cropped'] = align_info['cropped']
        info['alignment_info'] = align_info

    # Convert to numpy
    gen_arr = np.array(generated, dtype=np.float32)
    bg_arr = np.array(background, dtype=np.float32)

    # Compute mask
    mask = compute_difference_mask(
        gen_arr, bg_arr,
        threshold=threshold,
        feather_radius=feather_radius,
        diff_mode=diff_mode,
        min_alpha=min_alpha,
    )

    # Extract creature
    creature = extract_creature(gen_arr, bg_arr, mask, boost=boost)

    # Compute stats
    info['mask_coverage'] = float(np.mean(mask > 0.5) * 100)
    info['creature_max_brightness'] = int(creature.max())
    nonzero = creature[creature > 0]
    info['creature_mean_brightness'] = float(nonzero.mean()) if len(nonzero) > 0 else 0.0

    # Format output
    if output_mode == 'alpha':
        alpha_channel = (mask * 255).astype(np.uint8)
        creature_rgba = np.dstack([creature, alpha_channel])
        return creature_rgba, mask, info
    else:
        return creature, mask, info


def subtract_background_file(
    generated_path: Union[str, Path],
    background_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    threshold: Optional[int] = None,
    boost: Optional[float] = None,
    feather_radius: Optional[int] = None,
    diff_mode: Optional[DiffMode] = None,
    output_mode: Optional[OutputMode] = None,
    preview: bool = False,
) -> Tuple[Image.Image, dict]:
    """
    Load images and perform background subtraction.

    Args:
        generated_path: Path to generated frame (with creature)
        background_path: Path to background frame
        output_path: Path to save creature output (optional)
        threshold: Difference threshold (default from config)
        boost: Brightness boost (default from config)
        feather_radius: Edge feather radius (default from config)
        diff_mode: Difference calculation mode (default from config)
        output_mode: Output format (default from config)
        preview: Whether to also save a preview composite

    Returns:
        Tuple of (creature PIL Image, info dict)
    """
    # Apply defaults
    threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
    boost = boost if boost is not None else DEFAULT_BOOST
    feather_radius = feather_radius if feather_radius is not None else DEFAULT_FEATHER_RADIUS
    diff_mode = diff_mode if diff_mode is not None else DEFAULT_DIFF_MODE
    output_mode = output_mode if output_mode is not None else DEFAULT_OUTPUT_MODE
    min_alpha = DEFAULT_MIN_ALPHA
    aspect_tolerance = DEFAULT_ASPECT_TOLERANCE

    # Load images
    background = Image.open(background_path).convert('RGB')
    generated = Image.open(generated_path).convert('RGB')

    # Process
    creature_arr, mask, info = subtract_background(
        generated, background,
        threshold=threshold,
        boost=boost,
        feather_radius=feather_radius,
        diff_mode=diff_mode,
        min_alpha=min_alpha,
        output_mode=output_mode,
        aspect_tolerance=aspect_tolerance,
    )

    # Convert to PIL Image
    if output_mode == 'alpha':
        creature_img = Image.fromarray(creature_arr, mode='RGBA')
    else:
        creature_img = Image.fromarray(creature_arr)

    # Save outputs
    if output_path is not None:
        creature_img.save(output_path)
        info['output_path'] = str(output_path)

    if preview and output_path is not None:
        # Create preview composite
        bg_arr = np.array(background, dtype=np.float32)
        creature_rgb = creature_arr[:, :, :3] if output_mode == 'alpha' else creature_arr
        composite = np.clip(bg_arr + creature_rgb.astype(np.float32), 0, 255).astype(np.uint8)
        preview_img = Image.fromarray(composite)

        preview_path = Path(output_path).with_suffix('.preview.png')
        preview_img.save(preview_path)
        info['preview_path'] = str(preview_path)

    return creature_img, info


# ---------------------------------------------------------------------------
# CLI entry point: python -m wzrd.subtract
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}


def _cli():
    """CLI entry point for ``python -m wzrd.subtract``.

    Dispatches to subtract_video for video inputs, or processes a single image.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Background subtraction for projection mapping (image or video).',
    )
    parser.add_argument('input', help='Path to input image or video')
    parser.add_argument('background', help='Path to background frame image')
    parser.add_argument('-o', '--output', default=None,
                        help='Output path (default: auto-generated)')
    parser.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument('--boost', type=float, default=DEFAULT_BOOST)
    parser.add_argument('--feather', type=int, default=DEFAULT_FEATHER_RADIUS)
    parser.add_argument('--mode', choices=['rgb', 'lab', 'luminance'],
                        default=DEFAULT_DIFF_MODE)
    parser.add_argument('--output-mode', choices=['additive', 'alpha'],
                        default=DEFAULT_OUTPUT_MODE)
    parser.add_argument('--preview', action='store_true',
                        help='Also save a preview composite')
    # Video-only options
    parser.add_argument('--ramp', type=int, default=None,
                        help='Soft ramp width (video only, default: 20)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Gamma correction (video only, default: 0.85)')
    parser.add_argument('--morph-size', type=int, default=None,
                        help='Morphological cleanup kernel (video only)')
    parser.add_argument('--color-correction', type=float, default=None,
                        help='Color correction percentile (video only)')
    parser.add_argument('--temporal-smoothing', type=float, default=None,
                        help='Temporal EMA smoothing (video only)')
    parser.add_argument('--crf', type=int, default=18,
                        help='FFmpeg CRF quality (video only, default: 18)')
    args = parser.parse_args()

    input_path = Path(args.input)
    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS

    if is_video:
        from .subtract_video import subtract_background_video
        import time

        start_time = time.time()

        def progress(frame_num, total):
            if frame_num % 30 == 0 or frame_num == 1:
                total_str = str(total) if total else '?'
                elapsed = time.time() - start_time
                print(f'  Frame {frame_num}/{total_str}, time elapsed: {elapsed:.1f}s',
                      flush=True)

        print(f'Processing video: {args.input}')
        print(f'Background: {args.background}')

        # Build kwargs, only passing video-specific args if explicitly set
        kwargs = dict(
            output_path=args.output,
            threshold=args.threshold,
            feather_radius=args.feather,
            diff_mode=args.mode,
            output_mode=args.output_mode,
            preview=args.preview,
            crf=args.crf,
            progress_callback=progress,
        )
        if args.ramp is not None:
            kwargs['ramp'] = args.ramp
        if args.gamma is not None:
            kwargs['gamma'] = args.gamma
        if args.morph_size is not None:
            kwargs['morph_size'] = args.morph_size
        if args.color_correction is not None:
            kwargs['color_correction_percentile'] = args.color_correction
        if args.temporal_smoothing is not None:
            kwargs['temporal_smoothing'] = args.temporal_smoothing

        info = subtract_background_video(args.input, args.background, **kwargs)

        print(f'\nDone!')
        print(f'  Output: {info["output_video"]}')
        print(f'  Frames: {info["frames_processed"]}')
        if 'preview_video' in info:
            print(f'  Preview: {info["preview_video"]}')

    else:
        # Single image
        output = args.output
        if output is None:
            output = str(input_path.with_stem(input_path.stem + '_creature'))

        print(f'Processing image: {args.input}')
        print(f'Background: {args.background}')

        creature_img, info = subtract_background_file(
            args.input, args.background,
            output_path=output,
            threshold=args.threshold,
            boost=args.boost,
            feather_radius=args.feather,
            diff_mode=args.mode,
            output_mode=args.output_mode,
            preview=args.preview,
        )

        print(f'\nDone!')
        print(f'  Output: {info.get("output_path", output)}')
        print(f'  Mask coverage: {info["mask_coverage"]:.1f}%')
        print(f'  Max brightness: {info["creature_max_brightness"]}')


if __name__ == '__main__':
    _cli()
