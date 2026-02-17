"""
Background subtraction for creature extraction (single frame).

Isolates bright elements (creatures) from a dark background by computing
the difference between a generated frame and the original background.

Uses the same algorithmic pipeline as the video processor:
  1. Color correction (RGB luminance-based)
  2. Luminance difference at half resolution
  3. Soft ramp threshold
  4. Optional morphological opening
  5. Gaussian blur for smooth falloff
  6. Upscale mask to full resolution
  7. Creature extraction with gamma correction
"""

from PIL import Image
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
import cv2

from .utils import (
    align_to_reference, get_aspect_ratio, aspect_ratios_match,
    extract_creature, downscale_arr, upscale_arr, frac_to_px,
)

# Module-level defaults (matching video pipeline)
DEFAULT_THRESHOLD = 10
DEFAULT_RAMP = 20
DEFAULT_GAMMA = 0.85
DEFAULT_BLUR_RADIUS = 0.004         # fraction of min(H,W); ≈ 4 px at 1080p
DEFAULT_MORPH_SIZE = 0               # off by default (0 = disabled)
DEFAULT_ASPECT_TOLERANCE = 0.02
DEFAULT_COLOR_CORRECTION_THRESHOLD = 15  # max luminance diff to count as "background" pixel


def _luminance(arr: np.ndarray) -> np.ndarray:
    """Compute luminance from float32 RGB (HWC, 0-255). Returns HW float32."""
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _estimate_color_shift_frame(
    gen_arr: np.ndarray,
    bg_arr: np.ndarray,
    threshold: float = DEFAULT_COLOR_CORRECTION_THRESHOLD,
) -> np.ndarray:
    """Estimate per-channel RGB color shift for a single frame.

    Finds pixels where the luminance difference is below *threshold*
    (confident background pixels) and computes the mean per-channel shift.

    Args:
        gen_arr:   Generated frame as float32 (HWC, 0-255).
        bg_arr:    Background frame as float32 (HWC, 0-255).
        threshold: Max luminance difference for a pixel to count as
            "background" when computing the shift.

    Returns:
        3-element float32 RGB shift vector (to be added to bg_arr).
    """
    lum_diff = np.abs(_luminance(gen_arr) - _luminance(bg_arr))
    bg_mask = lum_diff < threshold

    if np.sum(bg_mask) < 100:
        return np.zeros(3, dtype=np.float32)

    shift = np.array([
        np.mean(gen_arr[bg_mask, 0] - bg_arr[bg_mask, 0]),
        np.mean(gen_arr[bg_mask, 1] - bg_arr[bg_mask, 1]),
        np.mean(gen_arr[bg_mask, 2] - bg_arr[bg_mask, 2]),
    ], dtype=np.float32)

    return shift


def subtract_background(
    generated: Union[Image.Image, np.ndarray],
    background: Union[Image.Image, np.ndarray],
    threshold: int = DEFAULT_THRESHOLD,
    ramp: int = DEFAULT_RAMP,
    gamma: float = DEFAULT_GAMMA,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    morph_size: float = DEFAULT_MORPH_SIZE,
    color_correction: bool = True,
    color_correction_threshold: float = DEFAULT_COLOR_CORRECTION_THRESHOLD,
    subtract_bg: bool = True,
    align: bool = True,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Subtract background from generated frame to isolate creature.

    Uses the same pipeline as the video processor:
      1. Color correction (RGB luminance-based shift applied to background)
      2. Luminance difference at half resolution
      3. Soft ramp threshold
      4. Optional morphological opening
      5. Gaussian blur for smooth falloff (max of original and blurred)
      6. Upscale mask to full resolution
      7. Creature extraction with gamma correction

    Args:
        generated:  Generated frame (PIL Image or numpy array).
        background: Background frame (PIL Image or numpy array).
        threshold:  Low cutoff for difference mask (0-255).
        ramp:       Soft transition width above threshold.
        gamma:      Gamma correction (< 1.0 brightens).
        blur_radius: Gaussian blur radius for mask feathering as fraction
            of min(H,W) (0 = disable).
        morph_size: Morphological kernel size as fraction of min(H,W)
            (0 = disable).
        color_correction: Whether to estimate and correct global color shift.
        color_correction_threshold: Max luminance diff for "background"
            pixels during color shift estimation.
        subtract_bg: Subtract background before masking (default True).
        align:      Whether to align generated to match background dims.
        aspect_tolerance: Tolerance for aspect ratio matching.

    Returns:
        Tuple of (creature_image, mask, info_dict).
    """
    info = {
        'threshold': threshold,
        'ramp': ramp,
        'gamma': gamma,
        'blur_radius': blur_radius,
        'morph_size': morph_size,
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

    h, w = gen_arr.shape[:2]

    # ---- Color correction (RGB luminance-based, same as video pipeline) ----
    if color_correction:
        color_shift = _estimate_color_shift_frame(
            gen_arr, bg_arr, threshold=color_correction_threshold,
        )
        if np.any(np.abs(color_shift) > 0.5):
            bg_arr = np.clip(bg_arr + color_shift, 0, 255)
            info['color_shift_rgb'] = color_shift.tolist()
        else:
            info['color_shift_rgb'] = [0.0, 0.0, 0.0]

    # ---- Downscale to half resolution for mask computation -----------------
    gen_arr_half = downscale_arr(gen_arr, 2)
    bg_arr_half = downscale_arr(bg_arr, 2)

    # ---- Luminance difference at half res ----------------------------------
    gen_lum_half = _luminance(gen_arr_half)
    bg_lum_half = _luminance(bg_arr_half)
    diff_mag = np.abs(gen_lum_half - bg_lum_half)

    # ---- Soft ramp threshold -----------------------------------------------
    ramp_width = max(float(ramp), 1e-6)
    mask = np.clip(
        (diff_mag - threshold) / ramp_width, 0, 1,
    ).astype(np.float32)

    # ---- Morphological opening (half res, off by default) ------------------
    half_ref = min(h // 2, w // 2)
    morph_px = frac_to_px(morph_size, half_ref, odd=True) if morph_size > 0 else 0
    if morph_px > 0:
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_px, morph_px),
        )
        mask_u8 = (mask * 255).astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kern)
        mask = mask_u8.astype(np.float32) / 255.0

    # ---- Gaussian blur for smooth falloff (half res) -----------------------
    blur_px = frac_to_px(blur_radius, half_ref) if blur_radius > 0 else 0
    blur_ksize = blur_px * 2 + 1 if blur_px > 0 else 0
    if blur_ksize > 0:
        blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
        mask = np.maximum(mask, blurred)

    # ---- Upscale mask to full res ------------------------------------------
    mask = upscale_arr(mask, (h, w))

    # ---- Extract creature --------------------------------------------------
    creature = extract_creature(gen_arr, bg_arr, mask, gamma=gamma,
                                subtract_bg=subtract_bg)

    # Compute stats
    info['mask_coverage'] = float(np.mean(mask > 0.5) * 100)
    info['creature_max_brightness'] = int(creature.max())
    nonzero = creature[creature > 0]
    info['creature_mean_brightness'] = (
        float(nonzero.mean()) if len(nonzero) > 0 else 0.0
    )

    return creature, mask, info


def subtract_background_file(
    generated_path: Union[str, Path],
    background_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    threshold: int = DEFAULT_THRESHOLD,
    ramp: int = DEFAULT_RAMP,
    gamma: float = DEFAULT_GAMMA,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    morph_size: float = DEFAULT_MORPH_SIZE,
    color_correction: bool = True,
    color_correction_threshold: float = DEFAULT_COLOR_CORRECTION_THRESHOLD,
    subtract_bg: bool = True,
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
        blur_radius:  Gaussian blur radius for mask feathering as fraction
            of min(H,W) (0 = disable).
        morph_size:   Morphological kernel size as fraction of min(H,W)
            (0 = disable).
        color_correction: Whether to estimate and correct color shift.
        color_correction_threshold: Max luminance diff for "background"
            pixels during color shift estimation.
        subtract_bg:  Subtract background before masking.
        preview:      Whether to also save a preview composite.

    Returns:
        Tuple of (creature PIL Image, info dict).
    """
    # Load images
    background = Image.open(background_path).convert('RGB')
    generated = Image.open(generated_path).convert('RGB')

    # Process
    creature_arr, mask, info = subtract_background(
        generated, background,
        threshold=threshold,
        ramp=ramp,
        gamma=gamma,
        blur_radius=blur_radius,
        morph_size=morph_size,
        color_correction=color_correction,
        color_correction_threshold=color_correction_threshold,
        subtract_bg=subtract_bg,
    )

    creature_img = Image.fromarray(creature_arr)

    # Save outputs
    if output_path is not None:
        creature_img.save(output_path)
        info['output_path'] = str(output_path)

    if preview and output_path is not None:
        bg_arr = np.array(background, dtype=np.float32)
        composite = np.clip(
            bg_arr + creature_arr.astype(np.float32), 0, 255
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
    parser.add_argument('--blur-radius', type=float, default=DEFAULT_BLUR_RADIUS,
                        help=f'Gaussian blur radius for mask feathering as fraction of min(H,W). Default: {DEFAULT_BLUR_RADIUS}')
    parser.add_argument('--morph-size', type=float, default=DEFAULT_MORPH_SIZE,
                        help=f'Morphological kernel as fraction of min(H,W) (0=off, default: {DEFAULT_MORPH_SIZE})')
    parser.add_argument('--no-color-correction', action='store_true',
                        help='Disable color shift estimation')
    parser.add_argument('--color-correction-threshold', type=float,
                        default=DEFAULT_COLOR_CORRECTION_THRESHOLD,
                        help=f'Max luminance diff for background pixels during color shift estimation (default: {DEFAULT_COLOR_CORRECTION_THRESHOLD})')
    parser.add_argument('--no-subtract-bg', action='store_true',
                        help='Disable background subtraction before masking (keeps raw projected colors)')
    parser.add_argument('--preview', action='store_true',
                        help='Also save a preview composite')
    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        input_path = Path(args.generated_frame)
        params_str = f"t{args.threshold}_g{args.gamma:.2f}_r{args.ramp}"
        args.output = str(input_path.with_name(f"{input_path.stem}_creature_{params_str}.png"))

    img, info = subtract_background_file(
        args.generated_frame,
        args.background_frame,
        output_path=args.output,
        threshold=args.threshold,
        ramp=args.ramp,
        gamma=args.gamma,
        blur_radius=args.blur_radius,
        morph_size=args.morph_size,
        color_correction=not args.no_color_correction,
        color_correction_threshold=args.color_correction_threshold,
        subtract_bg=not args.no_subtract_bg,
        preview=args.preview,
    )

    print(f"Saved: {info.get('output_path', args.output)}")
    print(f"  Mask coverage: {info['mask_coverage']:.1f}%")
    print(f"  Max brightness: {info['creature_max_brightness']}")
    if args.preview and 'preview_path' in info:
        print(f"  Preview: {info['preview_path']}")


if __name__ == '__main__':
    _cli()
