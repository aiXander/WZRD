"""
Background subtraction for creature extraction.

Functions to isolate bright elements (creatures) from a dark background
by computing the difference between generated frames and the original background.
"""

from PIL import Image
import numpy as np
from typing import Union, Optional, Tuple, Literal
from pathlib import Path

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
DEFAULT_ASPECT_TOLERANCE = 0.02


DiffMode = Literal['rgb', 'lab', 'luminance']
OutputMode = Literal['additive', 'alpha']


def subtract_background(
    generated: Union[Image.Image, np.ndarray],
    background: Union[Image.Image, np.ndarray],
    threshold: int = 10,
    ramp: int = 20,
    gamma: float = 0.85,
    feather_radius: int = 4,
    diff_mode: DiffMode = 'lab',
    min_alpha: float = 0.0,
    morph_size: int = 5,
    guided_filter_eps: float = 0.02,
    color_correction_percentile: float = 50,
    output_mode: OutputMode = 'additive',
    align: bool = True,
    aspect_tolerance: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Subtract background from generated frame to isolate creature.

    Args:
        generated:  Generated frame (PIL Image or numpy array).
        background: Background frame (PIL Image or numpy array).
        threshold:  Low cutoff for difference mask (0-255).
        ramp:       Soft transition width above threshold.
        gamma:      Gamma correction (< 1.0 brightens).
        feather_radius: Guided-filter feather radius in pixels.
        diff_mode:  ``'rgb'``, ``'lab'``, or ``'luminance'``.
        min_alpha:  Minimum mask value.
        morph_size: Morphological cleanup kernel size (0 = disable).
        guided_filter_eps: Guided-filter regularization.
        color_correction_percentile: Percentile for background pixel
            selection during color shift correction (0 = disable).
        output_mode: ``'additive'`` (RGB on black) or ``'alpha'`` (RGBA).
        align:      Whether to align generated to match background dims.
        aspect_tolerance: Tolerance for aspect ratio matching.

    Returns:
        Tuple of (creature_image, mask, info_dict).
    """
    info = {
        'threshold': threshold,
        'ramp': ramp,
        'gamma': gamma,
        'feather_radius': feather_radius,
        'diff_mode': diff_mode,
        'output_mode': output_mode,
        'morph_size': morph_size,
        'color_correction_percentile': color_correction_percentile,
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
        generated, align_info = align_to_reference(
            generated, background, aspect_tolerance
        )
        info['aligned'] = align_info['resized']
        info['cropped'] = align_info['cropped']
        info['alignment_info'] = align_info

    # Convert to numpy
    gen_arr = np.array(generated, dtype=np.float32)
    bg_arr = np.array(background, dtype=np.float32)

    # Color correction
    if color_correction_percentile > 0:
        gen_arr, color_shift, _ = correct_color_shift(
            gen_arr, bg_arr, color_correction_percentile
        )
        info['color_shift_lab'] = color_shift.tolist()

    # Compute mask
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

    # Extract creature
    creature = extract_creature(gen_arr, bg_arr, mask, gamma=gamma)

    # Compute stats
    info['mask_coverage'] = float(np.mean(mask > 0.5) * 100)
    info['creature_max_brightness'] = int(creature.max())
    nonzero = creature[creature > 0]
    info['creature_mean_brightness'] = (
        float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    )

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
    threshold: int = DEFAULT_THRESHOLD,
    ramp: int = DEFAULT_RAMP,
    gamma: float = DEFAULT_GAMMA,
    feather_radius: int = DEFAULT_FEATHER_RADIUS,
    diff_mode: DiffMode = DEFAULT_DIFF_MODE,
    output_mode: OutputMode = DEFAULT_OUTPUT_MODE,
    morph_size: int = DEFAULT_MORPH_SIZE,
    guided_filter_eps: float = DEFAULT_GUIDED_FILTER_EPS,
    color_correction_percentile: float = DEFAULT_COLOR_CORRECTION_PERCENTILE,
    preview: bool = False,
) -> Tuple[Image.Image, dict]:
    """
    Load images and perform background subtraction.

    Args:
        generated_path: Path to generated frame (with creature).
        background_path: Path to background frame.
        output_path:  Path to save creature output (optional).
        threshold:    Difference threshold (0-255).
        ramp:         Soft ramp width.
        gamma:        Gamma correction (< 1.0 brightens).
        feather_radius: Edge feather radius.
        diff_mode:    Difference mode ('rgb', 'lab', 'luminance').
        output_mode:  Output format ('additive' or 'alpha').
        morph_size:   Morphological kernel size (0 = disable).
        guided_filter_eps: Guided-filter regularization.
        color_correction_percentile: Color correction (0 = disable).
        preview:      Whether to also save a preview composite.

    Returns:
        Tuple of (creature PIL Image, info dict).
    """
    min_alpha = DEFAULT_MIN_ALPHA
    aspect_tolerance = DEFAULT_ASPECT_TOLERANCE

    # Load images
    background = Image.open(background_path).convert('RGB')
    generated = Image.open(generated_path).convert('RGB')

    # Process
    creature_arr, mask, info = subtract_background(
        generated, background,
        threshold=threshold,
        ramp=ramp,
        gamma=gamma,
        feather_radius=feather_radius,
        diff_mode=diff_mode,
        min_alpha=min_alpha,
        morph_size=morph_size,
        guided_filter_eps=guided_filter_eps,
        color_correction_percentile=color_correction_percentile,
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
        creature_rgb = (
            creature_arr[:, :, :3] if output_mode == 'alpha'
            else creature_arr
        )
        composite = np.clip(
            bg_arr + creature_rgb.astype(np.float32), 0, 255
        ).astype(np.uint8)
        preview_img = Image.fromarray(composite)

        preview_path = Path(output_path).with_suffix('.preview.png')
        preview_img.save(preview_path)
        info['preview_path'] = str(preview_path)

    return creature_img, info


def _cli():
    """CLI entry point for ``python -m wzrd.subtract_frame``."""
    import argparse

    parser = argparse.ArgumentParser(description='Subtract background from generated keyframe')
    parser.add_argument('generated_frame', help='Path to the AI-generated keyframe (with creature)')
    parser.add_argument('background_frame', help='Path to the original darkened background frame')
    parser.add_argument('-o', '--output', help='Output path (default: auto-generated)')
    parser.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD,
                        help=f'Difference threshold (0-255). Default: {DEFAULT_THRESHOLD}')
    parser.add_argument('--ramp', type=int, default=DEFAULT_RAMP,
                        help=f'Soft ramp width. Default: {DEFAULT_RAMP}')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                        help=f'Gamma correction (<1 brightens). Default: {DEFAULT_GAMMA}')
    parser.add_argument('--feather', type=int, default=DEFAULT_FEATHER_RADIUS,
                        help=f'Guided-filter feather radius. Default: {DEFAULT_FEATHER_RADIUS}')
    parser.add_argument('--mode', type=str, choices=['rgb', 'lab', 'luminance'],
                        default=DEFAULT_DIFF_MODE,
                        help=f'Difference calculation mode. Default: {DEFAULT_DIFF_MODE}')
    parser.add_argument('--output-mode', type=str, choices=['additive', 'alpha'],
                        default=DEFAULT_OUTPUT_MODE,
                        help=f'Output format. Default: {DEFAULT_OUTPUT_MODE}')
    parser.add_argument('--color-correction', type=float,
                        default=DEFAULT_COLOR_CORRECTION_PERCENTILE,
                        help=f'Color correction percentile (0=off). Default: {DEFAULT_COLOR_CORRECTION_PERCENTILE}')
    parser.add_argument('--preview', action='store_true',
                        help='Also save a preview composite')
    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.generated_frame)
        params_str = f"t{args.threshold}_g{args.gamma:.2f}_f{args.feather}_{args.mode}"
        suffix = '_alpha.png' if args.output_mode == 'alpha' else '.png'
        args.output = str(input_path.with_name(f"{input_path.stem}_creature_{params_str}{suffix}"))

    img, info = subtract_background_file(
        args.generated_frame,
        args.background_frame,
        output_path=args.output,
        threshold=args.threshold,
        ramp=args.ramp,
        gamma=args.gamma,
        feather_radius=args.feather,
        diff_mode=args.mode,
        output_mode=args.output_mode,
        color_correction_percentile=args.color_correction,
        preview=args.preview,
    )

    print(f"Saved: {info.get('output_path', args.output)}")
    print(f"  Mask coverage: {info['mask_coverage']:.1f}%")
    print(f"  Max brightness: {info['creature_max_brightness']}")
    if args.preview and 'preview_path' in info:
        print(f"  Preview: {info['preview_path']}")


if __name__ == '__main__':
    _cli()
