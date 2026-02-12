"""
Image darkening for projection mapping.

Uses Gradient-Weighted Histogram Equalization to allocate the available
luminance budget based on texture density (gradients), not pixel count.
Flat regions (sky, shadows, blank walls) get near-zero allocation;
textured regions (bricks, edges, architectural detail) get maximally
expanded dynamic range within the budget.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple
from pathlib import Path

from .utils import (
    parse_aspect_ratio,
    normalize_image,
    compute_target_dimensions,
    DebugContext,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_BRIGHTNESS = 0.25       # Luminance budget ceiling (0.0–1.0)
DETAIL_BOOST = 1.15         # Detail layer amplification factor
GUIDE_FRACTION = 0.04       # Guided filter radius as fraction of min(H, W)
GUIDE_EPS = 0.01            # Guided filter regularization (relative to image dynamic range)
HIST_BINS = 256             # CDF histogram resolution
CHROMA_CORRECTION = 0.5     # Chroma scaling strength (0=none, 1=full)
GRADIENT_SOURCE = 0.5       # 0=raw-L gradients, 1=detail-residual gradients
DEFAULT_ASPECT = '16:9'
BASE_RESOLUTION = 1920

# ---------------------------------------------------------------------------
# Step 1: Gradient Magnitude (Information Map)
# ---------------------------------------------------------------------------

def _gradient_magnitude(L: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude of the luminance channel.

    Uses Scharr operators for better rotational symmetry than Sobel.

    Args:
        L: Luminance channel (float32, any range)

    Returns:
        Gradient magnitude map (float32, >= 0)
    """
    dx = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(L, cv2.CV_32F, 0, 1)
    return np.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Steps 2–3: Gradient-Weighted CDF Tone Curve
# ---------------------------------------------------------------------------

def _gradient_weighted_cdf(
    L: np.ndarray,
    grad: np.ndarray,
    bins: int = HIST_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a CDF of luminance weighted by gradient magnitude.

    Standard histogram equalization weights every pixel equally, so a
    million flat-sky pixels dominate the allocation.  Here we weight
    each pixel's contribution by its gradient magnitude, giving zero
    weight to featureless regions and full weight to textured ones.

    Args:
        L:    Luminance values (float32)
        grad: Gradient magnitude per pixel (float32)
        bins: Number of histogram bins

    Returns:
        (bin_centers, cdf) where cdf is normalised to [0, 1]
    """
    L_flat = L.ravel()
    G_flat = grad.ravel().astype(np.float64)

    L_min, L_max = float(L_flat.min()), float(L_flat.max())
    if L_max - L_min < 1e-8:
        # Degenerate: constant-luminance image
        return np.array([L_min], dtype=np.float32), np.array([1.0], dtype=np.float32)

    edges = np.linspace(L_min, L_max, bins + 1, dtype=np.float64)
    indices = np.clip(np.digitize(L_flat, edges) - 1, 0, bins - 1)

    # Weighted histogram: sum gradient magnitudes per bin, not pixel counts
    weighted_hist = np.bincount(indices, weights=G_flat, minlength=bins)[:bins]

    # CDF
    cdf = np.cumsum(weighted_hist)
    total = cdf[-1]
    if total > 0:
        cdf /= total
    else:
        # Zero gradient everywhere → uniform mapping fallback
        cdf = np.linspace(0, 1, bins)

    centers = (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)
    return centers, cdf.astype(np.float32)


def _apply_tone_curve(
    L: np.ndarray,
    centers: np.ndarray,
    cdf: np.ndarray,
    max_brightness: float,
) -> np.ndarray:
    """
    Map luminance through the gradient-weighted CDF and scale to budget.

    Regions where the CDF is flat (no gradient density) get compressed;
    regions where the CDF rises steeply get expanded dynamic range.

    Args:
        L:              Input luminance (float32)
        centers:        Histogram bin centers
        cdf:            Normalised CDF [0, 1]
        max_brightness: Output luminance ceiling

    Returns:
        Remapped luminance in [0, max_brightness]
    """
    mapped = np.interp(L, centers, cdf).astype(np.float32)
    return mapped * max_brightness


# ---------------------------------------------------------------------------
# Step 4: Guided Filter (Base / Detail Separation)
# ---------------------------------------------------------------------------

def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """
    Edge-preserving smoothing via guided filter.

    Superior to bilateral filter for base/detail decomposition because
    it prevents gradient-reversal artifacts at strong edges.

    Args:
        guide:  Guidance image (float32, single channel)
        src:    Source image to filter (float32, single channel)
        radius: Filter window radius in pixels
        eps:    Regularization (larger → smoother base)

    Returns:
        Filtered (base) image (float32)
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    return mean_a * guide + mean_b


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def darken_image(
    image: Union[Image.Image, np.ndarray],
    max_brightness: float = MAX_BRIGHTNESS,
    detail_boost: float = DETAIL_BOOST,
    guide_fraction: float = GUIDE_FRACTION,
    guide_eps: float = GUIDE_EPS,
    hist_bins: int = HIST_BINS,
    chroma_correction: float = CHROMA_CORRECTION,
    gradient_source: float = GRADIENT_SOURCE,
    debug: Optional[DebugContext] = None,
) -> np.ndarray:
    """
    Darken an image for projection mapping.

    Uses Gradient-Weighted Histogram Equalization:

    1. Convert to CIELAB; decompose L into base + detail via guided
       filter (eps scaled to image dynamic range for input invariance).
    2. Compute gradient magnitude as a blend of raw-L and
       detail-residual gradients (controlled by ``gradient_source``).
       This lets dark textured regions receive proper weight even when
       their absolute gradients are tiny.
    3. Build luminance histogram weighted by the blended gradient
       (not pixel count).  Flat regions contribute nothing; textured
       regions dominate.
    4. Derive an adaptive tone curve from the weighted CDF; apply to
       the base layer only, scaled to the luminance budget.
    5. Soft-compress detail outliers via tanh knee (scale = 95th
       percentile of |detail|), then recombine with boosted detail
       and clip to the luminance budget.

    Args:
        image:          PIL Image or numpy array (HWC, 0-255)
        max_brightness: Output luminance ceiling (0.0-1.0).
        detail_boost:   Detail layer amplification (1=preserve, >1=enhance).
        guide_fraction: Guided filter radius as fraction of min(H, W).
        guide_eps:      Guided filter regularisation, relative to image
                        dynamic range (larger = smoother base).
        hist_bins:      Histogram bins for the gradient-weighted CDF.
        chroma_correction: How strongly to scale chroma with luminance
                        reduction (0.0=keep original colour, 1.0=full
                        proportional reduction). Default 0.5.
        gradient_source: Blend between raw-L gradients (0.0) and
                        detail-residual gradients (1.0). Default 0.5.
                        Higher values give more weight to texture in
                        dark regions; lower values preserve structural
                        edge weighting.

    Returns:
        Darkened image as numpy array (uint8, 0-255)
    """
    # --- Input normalisation → uint8 RGB ---
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert('RGB'), dtype=np.uint8)
    else:
        arr = image.astype(np.float32)
        if arr.max() <= 1.0:
            rgb = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(arr, 0, 255).astype(np.uint8)

    if debug is None:
        debug = DebugContext()

    # --- Step 1: RGB → CIELAB, extract L channel ---
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32) / 255.0  # Normalise to [0, 1]
    debug.save("input_rgb", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    debug.save("luminance_L", L)

    # --- Step 2: Guided filter → base + detail ---
    # Scale eps to the image's effective dynamic range so that the
    # filter behaves consistently regardless of overall brightness.
    # For a full-range image (range≈1) eps is unchanged; for a dark
    # image (range≈0.2) eps shrinks proportionally so texture in dark
    # regions is still detected as edges rather than smoothed away.
    radius = max(1, int(min(rgb.shape[:2]) * guide_fraction))
    L_range = np.percentile(L, 97.5) - np.percentile(L, 2.5) + 1e-8
    eps_scaled = guide_eps * (L_range ** 2)
    base = _guided_filter(L, L, radius, eps_scaled)
    detail = L - base  # Signed residual
    debug.save("base_guided_filter", base)
    debug.save("detail_residual", detail)

    # --- Step 3: Blended gradient magnitude ---
    # Raw-L gradients capture structural edges but under-weight dark
    # textured regions.  Detail-residual gradients capture texture
    # regardless of absolute brightness but miss structural edges
    # (which live in the base layer).  Blend the two for best of both.
    grad_raw = _gradient_magnitude(L)
    grad_detail = _gradient_magnitude(detail)
    # Normalise each to [0, 1] so the blend is scale-invariant
    grad_raw_n = grad_raw / (grad_raw.max() + 1e-8)
    grad_detail_n = grad_detail / (grad_detail.max() + 1e-8)
    grad = (1.0 - gradient_source) * grad_raw_n + gradient_source * grad_detail_n
    debug.save("gradient_magnitude", grad)

    # --- Step 4: Gradient-weighted CDF ---
    centers, cdf = _gradient_weighted_cdf(L, grad, bins=hist_bins)

    # --- Apply tone curve to base, scale to budget ---
    compressed_base = _apply_tone_curve(base, centers, cdf, max_brightness)
    debug.save("compressed_base", compressed_base)

    # --- Step 5: Soft-compress detail outliers (tanh knee) ---
    # Scale derived from the detail layer itself so it's image-adaptive.
    # Small details (|d| << scale) pass through ~linearly;
    # bright-light spikes get smoothly squashed toward ±scale.
    detail_scale = np.percentile(np.abs(detail), 95) + 1e-8
    detail_compressed = detail_scale * np.tanh(detail / detail_scale)
    debug.save("detail_compressed", detail_compressed)

    # --- Step 6: Recombine with boosted detail, enforce budget ---
    new_L = np.clip(
        compressed_base + detail_compressed * detail_boost,
        0.0,
        max_brightness,
    )
    debug.save("recombined_L", new_L)

    # --- Scale chroma proportionally to luminance reduction ---
    # Use the base-layer ratio (smooth) rather than full per-pixel ratio,
    # so high-frequency detail doesn't cause noisy chroma swings.
    L_ratio = compressed_base / (base + 0.01 * L_range)  # smooth darkening ratio
    L_ratio = np.minimum(L_ratio, 1.0)

    a_centered = lab[..., 1].astype(np.float32) - 128.0
    b_centered = lab[..., 2].astype(np.float32) - 128.0

    # Weight correction by original chroma: saturated pixels need more
    # correction to avoid garish colours; near-neutral pixels need almost
    # none to avoid graying out.  C*=50 in CIELAB is "moderately saturated".
    chroma_orig = np.sqrt(a_centered ** 2 + b_centered ** 2)
    chroma_weight = np.clip(chroma_orig / 50.0, 0.0, 1.0)

    # Blend: identity (1.0) → full correction (L_ratio), modulated by
    # both the user strength knob and how chromatic the pixel is.
    effective_ratio = 1.0 - chroma_correction * chroma_weight * (1.0 - L_ratio)

    a_new = a_centered * effective_ratio
    b_new = b_centered * effective_ratio

    # Soft-clip a,b into valid range using tanh knee.
    # Values below 90% of the limit pass through ~linearly;
    # values beyond that get smoothly compressed instead of hard-clipped.
    ab_limit = 127.0
    ab_knee = 0.9
    ab_thresh = ab_limit * ab_knee
    ab_headroom = ab_limit - ab_thresh
    for ch in (a_new, b_new):
        abs_ch = np.abs(ch)
        excess = abs_ch - ab_thresh
        mask = excess > 0
        if mask.any():
            ch[mask] = np.sign(ch[mask]) * (
                ab_thresh + ab_headroom * np.tanh(excess[mask] / ab_headroom)
            )

    # --- Reconstruct LAB → RGB ---
    out_lab = lab.copy()
    out_lab[..., 0] = np.clip(np.round(new_L * 255.0), 0, 255).astype(np.uint8)
    out_lab[..., 1] = np.round(128.0 + a_new).astype(np.uint8)
    out_lab[..., 2] = np.round(128.0 + b_new).astype(np.uint8)

    result = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    debug.save("final_output", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return result


# ---------------------------------------------------------------------------
# File-level API
# ---------------------------------------------------------------------------

def darken_image_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    max_brightness: float = MAX_BRIGHTNESS,
    detail_boost: float = DETAIL_BOOST,
    guide_fraction: float = GUIDE_FRACTION,
    guide_eps: float = GUIDE_EPS,
    hist_bins: int = HIST_BINS,
    chroma_correction: float = CHROMA_CORRECTION,
    gradient_source: float = GRADIENT_SOURCE,
    target_aspect: str = DEFAULT_ASPECT,
    base_resolution: int = BASE_RESOLUTION,
    normalize: bool = True,
    debug_dir: Optional[str] = None,
) -> Tuple[Image.Image, dict]:
    """
    Load, optionally normalise, and darken an image file.

    Args:
        input_path:     Path to input image
        output_path:    Path to save output (optional)
        max_brightness: Output luminance ceiling (0.0-1.0)
        detail_boost:   Detail layer amplification
        guide_fraction: Guided filter radius as fraction of min(H, W)
        guide_eps:      Guided filter regularisation (relative to dynamic range)
        hist_bins:      Histogram resolution for CDF
        chroma_correction: Chroma scaling strength (0=none, 1=full)
        gradient_source: Blend raw-L (0) vs detail-residual (1) gradients
        target_aspect:  Aspect ratio string e.g. "16:9"
        base_resolution: Max dimension in pixels
        normalize:      Whether to normalise aspect ratio and resolution

    Returns:
        Tuple of (darkened PIL Image, info dict with processing details)
    """
    input_path = Path(input_path)
    img = Image.open(input_path).convert('RGB')
    original_size = img.size

    info = {
        'original_size': original_size,
        'max_brightness': max_brightness,
        'detail_boost': detail_boost,
        'guide_fraction': guide_fraction,
        'guide_eps': guide_eps,
        'hist_bins': hist_bins,
        'chroma_correction': chroma_correction,
        'gradient_source': gradient_source,
        'normalized': False,
        'target_aspect': None,
        'final_size': None,
    }

    if normalize:
        target_aspect_val = parse_aspect_ratio(target_aspect)
        img = normalize_image(img, target_aspect_val, base_resolution)
        info['normalized'] = True
        info['target_aspect'] = target_aspect

    info['final_size'] = img.size

    debug = DebugContext(debug_dir)
    darkened_arr = darken_image(
        img,
        max_brightness=max_brightness,
        detail_boost=detail_boost,
        guide_fraction=guide_fraction,
        guide_eps=guide_eps,
        hist_bins=hist_bins,
        chroma_correction=chroma_correction,
        gradient_source=gradient_source,
        debug=debug,
    )
    darkened_img = Image.fromarray(darkened_arr)

    if output_path is not None:
        darkened_img.save(output_path, quality=95)
        info['output_path'] = str(output_path)

    return darkened_img, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    """CLI entry point for `python -m wzrd.darken`."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Darken an image for projection mapping '
                    '(Gradient-Weighted Histogram Equalization)'
    )
    parser.add_argument('input_image', help='Path to the input image')
    parser.add_argument('-o', '--output', help='Output path (default: auto-generated)')
    parser.add_argument('--max-brightness', type=float, default=MAX_BRIGHTNESS,
                        help=f'Luminance budget ceiling (0.0-1.0). Default: {MAX_BRIGHTNESS}')
    parser.add_argument('--detail-boost', type=float, default=DETAIL_BOOST,
                        help=f'Detail layer amplification (1=preserve, >1=enhance). '
                             f'Default: {DETAIL_BOOST}')
    parser.add_argument('--guide-fraction', type=float, default=GUIDE_FRACTION,
                        help=f'Guided filter radius as fraction of image size. '
                             f'Default: {GUIDE_FRACTION}')
    parser.add_argument('--guide-eps', type=float, default=GUIDE_EPS,
                        help=f'Guided filter regularisation. Default: {GUIDE_EPS}')
    parser.add_argument('--hist-bins', type=int, default=HIST_BINS,
                        help=f'CDF histogram resolution. Default: {HIST_BINS}')
    parser.add_argument('--chroma-correction', type=float, default=CHROMA_CORRECTION,
                        help=f'Chroma scaling strength (0=none, 1=full). '
                             f'Default: {CHROMA_CORRECTION}')
    parser.add_argument('--gradient-source', type=float, default=GRADIENT_SOURCE,
                        help=f'Blend raw-L gradients (0) vs detail-residual (1). '
                             f'Higher values reveal texture in dark regions. '
                             f'Default: {GRADIENT_SOURCE}')
    parser.add_argument('--aspect', type=str, default=DEFAULT_ASPECT,
                        help=f'Target aspect ratio (e.g., 16:9). Default: {DEFAULT_ASPECT}')
    parser.add_argument('--base-resolution', type=int, default=BASE_RESOLUTION,
                        help=f'Base resolution. Default: {BASE_RESOLUTION}')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip aspect ratio and resolution normalization')
    parser.add_argument('--debug', action='store_true',
                        help='Save intermediate debug images next to the input')
    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input_image)
        args.output = str(input_path.with_name(f"{input_path.stem}_darkened.jpg"))

    debug_dir = str(Path(args.input_image).parent) if args.debug else None

    img, info = darken_image_file(
        args.input_image,
        output_path=args.output,
        max_brightness=args.max_brightness,
        detail_boost=args.detail_boost,
        guide_fraction=args.guide_fraction,
        guide_eps=args.guide_eps,
        hist_bins=args.hist_bins,
        chroma_correction=args.chroma_correction,
        gradient_source=args.gradient_source,
        target_aspect=args.aspect,
        base_resolution=args.base_resolution,
        normalize=not args.no_normalize,
        debug_dir=debug_dir,
    )

    print(f"Saved: {info.get('output_path', args.output)}")
    print(f"  Size: {info['final_size'][0]}x{info['final_size'][1]}")
    print(f"  Max brightness: {info['max_brightness']}")
    print(f"  Detail boost: {info['detail_boost']}")
    print(f"  Guide: fraction={info['guide_fraction']}, eps={info['guide_eps']}")


if __name__ == '__main__':
    _cli()
