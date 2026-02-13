"""
Utility functions for projection mapping pipeline.

Handles resolution normalization, aspect ratio cropping, and image alignment
to ensure consistency across AI-generated content.
"""

from PIL import Image
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import cv2
import os

# Standard aspect ratios (width:height)
ASPECT_RATIOS = {
    '16:9': 16 / 9,
    '9:16': 9 / 16,
    '4:3': 4 / 3,
    '3:4': 3 / 4,
    '1:1': 1.0,
    '21:9': 21 / 9,
    '9:21': 9 / 21,
}

# Default base resolution (max dimension)
DEFAULT_BASE_RESOLUTION = 1920

# Tolerance for aspect ratio matching (percentage)
ASPECT_RATIO_TOLERANCE = 0.02  # 2% tolerance



class DebugContext:
    """Context for saving debug images with numbered prefixes."""

    def __init__(self, debug_dir: Optional[str] = None):
        self.debug_dir = debug_dir
        self.step = 0
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.debug_dir is not None

    def save(self, name: str, image: np.ndarray, normalize: bool = False):
        """Save a debug image with numbered prefix."""
        if not self.enabled:
            return

        self.step += 1

        # Handle different image types
        if image is None:
            return

        img_to_save = image.copy()

        # Normalize float images to 0-255 range
        if normalize or img_to_save.dtype in [np.float32, np.float64]:
            img_to_save = cv2.normalize(img_to_save, None, 0, 255, cv2.NORM_MINMAX)
            img_to_save = img_to_save.astype(np.uint8)

        filename = f"{self.step:02d}_{name}.png"
        filepath = os.path.join(self.debug_dir, filename)
        cv2.imwrite(filepath, img_to_save)
        print(f"  [debug] Saved: {filename}")

    def save_with_corners(self, name: str, image: np.ndarray, corners: np.ndarray,
                          color: Tuple[int, int, int] = (0, 0, 255)):
        """Save image with corners drawn on it."""
        if not self.enabled or corners is None:
            return

        self.step += 1

        # Make sure we have a color image
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Normalize if needed
        if vis.dtype in [np.float32, np.float64]:
            vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Draw polygon
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, color, 2)

        # Draw corner points with labels
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, corner in enumerate(corners):
            pt = (int(corner[0]), int(corner[1]))
            cv2.circle(vis, pt, 6, color, -1)
            cv2.circle(vis, pt, 8, (255, 255, 255), 1)
            cv2.putText(vis, labels[i], (pt[0] + 10, pt[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        filename = f"{self.step:02d}_{name}.png"
        filepath = os.path.join(self.debug_dir, filename)
        cv2.imwrite(filepath, vis)
        print(f"  [debug] Saved: {filename}")



def parse_aspect_ratio(aspect_str: str) -> float:
    """
    Parse aspect ratio string to float.

    Args:
        aspect_str: Aspect ratio string like "16:9" or "16x9"

    Returns:
        Float aspect ratio (width/height)
    """
    if aspect_str in ASPECT_RATIOS:
        return ASPECT_RATIOS[aspect_str]

    # Try parsing custom format
    for sep in [':', 'x', '/']:
        if sep in aspect_str:
            parts = aspect_str.split(sep)
            if len(parts) == 2:
                try:
                    w, h = float(parts[0]), float(parts[1])
                    return w / h
                except ValueError:
                    pass

    raise ValueError(f"Invalid aspect ratio format: {aspect_str}. Use format like '16:9'")


def get_aspect_ratio(size: Tuple[int, int]) -> float:
    """Get aspect ratio from (width, height) tuple."""
    return size[0] / size[1]


def aspect_ratios_match(ar1: float, ar2: float, tolerance: float = ASPECT_RATIO_TOLERANCE) -> bool:
    """Check if two aspect ratios match within tolerance."""
    return abs(ar1 - ar2) / max(ar1, ar2) <= tolerance


def get_closest_standard_aspect(aspect: float) -> Tuple[str, float]:
    """
    Find the closest standard aspect ratio.

    Returns:
        Tuple of (aspect_name, aspect_value)
    """
    closest_name = None
    closest_diff = float('inf')

    for name, value in ASPECT_RATIOS.items():
        diff = abs(aspect - value)
        if diff < closest_diff:
            closest_diff = diff
            closest_name = name

    return closest_name, ASPECT_RATIOS[closest_name]


def center_crop_to_aspect(
    image: Image.Image,
    target_aspect: float
) -> Image.Image:
    """
    Center crop an image to match target aspect ratio.

    Args:
        image: PIL Image to crop
        target_aspect: Target aspect ratio (width/height)

    Returns:
        Cropped PIL Image
    """
    img_width, img_height = image.size
    current_aspect = img_width / img_height

    if aspect_ratios_match(current_aspect, target_aspect):
        return image  # Already correct aspect ratio

    if current_aspect > target_aspect:
        # Image is wider than target: crop width
        new_width = int(img_height * target_aspect)
        new_height = img_height
    else:
        # Image is taller than target: crop height
        new_width = img_width
        new_height = int(img_width / target_aspect)

    # Calculate center crop coordinates
    left = (img_width - new_width) // 2
    top = (img_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


def resize_to_base_resolution(
    image: Image.Image,
    base_resolution: int = DEFAULT_BASE_RESOLUTION,
    resample: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image so max dimension equals base_resolution, maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        base_resolution: Target max dimension
        resample: PIL resampling method

    Returns:
        Resized PIL Image
    """
    width, height = image.size
    max_dim = max(width, height)

    if max_dim == base_resolution:
        return image

    scale = base_resolution / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), resample)


def normalize_image(
    image: Image.Image,
    target_aspect: float,
    base_resolution: int = DEFAULT_BASE_RESOLUTION
) -> Image.Image:
    """
    Normalize image to standard aspect ratio and base resolution.

    This is the main function for preparing the initial surface image.
    Output guarantees: max(w,h) = base_resolution and exact target aspect ratio.

    Args:
        image: PIL Image to normalize
        target_aspect: Target aspect ratio (width/height)
        base_resolution: Target max dimension

    Returns:
        Normalized PIL Image
    """
    # Compute exact target dimensions for precise aspect ratio with max(w,h) = base_resolution
    target_width, target_height = compute_target_dimensions(target_aspect, base_resolution)

    # Center crop to target aspect ratio (minimal crop)
    cropped = center_crop_to_aspect(image, target_aspect)

    # Resize to exact target dimensions to guarantee both aspect ratio and max dimension
    resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return resized


def align_to_reference(
    image: Image.Image,
    reference: Image.Image,
    tolerance: float = ASPECT_RATIO_TOLERANCE
) -> Tuple[Image.Image, dict]:
    """
    Align an image to match a reference image's dimensions.

    Handles common AI generator issues:
    - Slightly different resolutions -> resize to match
    - Different aspect ratios -> center crop then resize

    Args:
        image: Image to align (e.g., AI-generated frame)
        reference: Reference image to match (e.g., darkened background)
        tolerance: Aspect ratio tolerance for warnings

    Returns:
        Tuple of (aligned_image, info_dict)
        info_dict contains: 'resized', 'cropped', 'original_size', 'warnings'
    """
    ref_size = reference.size
    img_size = image.size

    info = {
        'resized': False,
        'cropped': False,
        'original_size': img_size,
        'final_size': ref_size,
        'warnings': []
    }

    # Check if already matching
    if img_size == ref_size:
        return image, info

    ref_aspect = get_aspect_ratio(ref_size)
    img_aspect = get_aspect_ratio(img_size)

    result = image

    # Check aspect ratio mismatch
    if not aspect_ratios_match(img_aspect, ref_aspect, tolerance):
        info['warnings'].append(
            f"Aspect ratio mismatch: image={img_aspect:.3f}, reference={ref_aspect:.3f}. "
            f"Center cropping to match."
        )
        result = center_crop_to_aspect(result, ref_aspect)
        info['cropped'] = True

    # Resize to match reference dimensions
    if result.size != ref_size:
        info['resized'] = True
        result = result.resize(ref_size, Image.Resampling.LANCZOS)

    return result, info


def align_array_to_reference(
    array: np.ndarray,
    reference_size: Tuple[int, int],
    tolerance: float = ASPECT_RATIO_TOLERANCE
) -> Tuple[np.ndarray, dict]:
    """
    Align a numpy array (HWC format) to match reference dimensions.

    Args:
        array: NumPy array in HWC format
        reference_size: Target (width, height)
        tolerance: Aspect ratio tolerance

    Returns:
        Tuple of (aligned_array, info_dict)
    """
    # Convert to PIL for processing
    image = Image.fromarray(array.astype(np.uint8))

    # Create a dummy reference image for alignment
    ref_image = Image.new('RGB', reference_size)

    aligned, info = align_to_reference(image, ref_image, tolerance)

    return np.array(aligned), info


def load_and_normalize(
    path: Union[str, Path],
    target_aspect: Optional[float] = None,
    base_resolution: Optional[int] = None
) -> Image.Image:
    """
    Load an image and optionally normalize it.

    Args:
        path: Path to image file
        target_aspect: If provided, crop to this aspect ratio
        base_resolution: If provided, resize to this max dimension

    Returns:
        PIL Image (RGB)
    """
    image = Image.open(path).convert('RGB')

    if target_aspect is not None:
        image = center_crop_to_aspect(image, target_aspect)

    if base_resolution is not None:
        image = resize_to_base_resolution(image, base_resolution)

    return image


def load_and_align(
    path: Union[str, Path],
    reference: Image.Image,
    tolerance: float = ASPECT_RATIO_TOLERANCE
) -> Tuple[Image.Image, dict]:
    """
    Load an image and align it to a reference.

    Args:
        path: Path to image file
        reference: Reference image to match
        tolerance: Aspect ratio tolerance

    Returns:
        Tuple of (aligned_image, info_dict)
    """
    image = Image.open(path).convert('RGB')
    return align_to_reference(image, reference, tolerance)


def compute_target_dimensions(
    target_aspect: float,
    base_resolution: int = DEFAULT_BASE_RESOLUTION
) -> Tuple[int, int]:
    """
    Compute target (width, height) for a given aspect ratio and base resolution.

    Args:
        target_aspect: Target aspect ratio (width/height)
        base_resolution: Max dimension

    Returns:
        Tuple of (width, height)
    """
    if target_aspect >= 1.0:
        # Landscape or square: width is the max dimension
        width = base_resolution
        height = round(base_resolution / target_aspect)
    else:
        # Portrait: height is the max dimension
        height = base_resolution
        width = round(base_resolution * target_aspect)

    return width, height


def print_alignment_info(info: dict, prefix: str = "  ") -> None:
    """Print alignment info in a readable format."""
    if info['cropped'] or info['resized']:
        print(f"{prefix}Original size: {info['original_size']}")
        print(f"{prefix}Final size: {info['final_size']}")
        if info['cropped']:
            print(f"{prefix}Applied center crop to match aspect ratio")
        if info['resized']:
            print(f"{prefix}Resized to match reference dimensions")

    for warning in info.get('warnings', []):
        print(f"{prefix}Warning: {warning}")


# ---------------------------------------------------------------------------
# Color space conversion
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float32 (0-255) to CIELab float32 via OpenCV.

    Returns L in [0, 100], a and b in [-127, 127].
    """
    rgb_01 = np.clip(rgb / 255.0, 0, 1).astype(np.float32)
    bgr = cv2.cvtColor(rgb_01, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELab float32 to RGB float32 (0-255) via OpenCV."""
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    rgb_01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.clip(rgb_01 * 255.0, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Guided filter
# ---------------------------------------------------------------------------

def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Edge-preserving guided filter (O(N) box-filter implementation).

    Smooths *src* while preserving edges present in *guide*.

    Args:
        guide: Guide image (HW, float32, typically 0-1).
        src:   Source image to filter (HW, float32).
        radius: Filter radius in pixels.
        eps:   Regularization (larger → smoother, smaller → more edge-preserving).

    Returns:
        Filtered image (HW, float32).
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_g = cv2.boxFilter(guide, -1, ksize)
    mean_s = cv2.boxFilter(src, -1, ksize)
    corr_gs = cv2.boxFilter(guide * src, -1, ksize)
    corr_gg = cv2.boxFilter(guide * guide, -1, ksize)

    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    return mean_a * guide + mean_b


# ---------------------------------------------------------------------------
# Color shift correction
# ---------------------------------------------------------------------------

def correct_color_shift(
    generated: np.ndarray,
    background: np.ndarray,
    percentile: float = 50,
    *,
    gen_lab: Optional[np.ndarray] = None,
    bg_lab: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct global color/brightness shift in perceptual CIELab space.

    AI video models often introduce a systematic color offset from the
    reference image.  This estimates the offset using *confident background
    pixels* (those with the smallest per-pixel difference) and subtracts it.

    Args:
        generated:  Generated frame float32 (HWC, 0-255).
        background: Background frame float32 (HWC, 0-255).
        percentile: Use lowest N% of difference pixels as reference.

    Returns:
        Tuple of (corrected_rgb_float32, lab_shift_vector, corrected_lab).
    """
    if gen_lab is None:
        gen_lab = rgb_to_lab(generated)
    if bg_lab is None:
        bg_lab = rgb_to_lab(background)

    # Per-pixel LAB Euclidean distance (first-pass rough difference)
    diff = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))

    # Select confident background pixels
    cutoff = np.percentile(diff, percentile)
    bg_mask = diff <= cutoff

    if np.sum(bg_mask) < 100:
        # Not enough background pixels — skip correction
        return generated.copy(), np.zeros(3, dtype=np.float32), gen_lab

    # Average LAB shift across background pixels
    shift = np.array([
        np.mean(gen_lab[bg_mask, 0] - bg_lab[bg_mask, 0]),
        np.mean(gen_lab[bg_mask, 1] - bg_lab[bg_mask, 1]),
        np.mean(gen_lab[bg_mask, 2] - bg_lab[bg_mask, 2]),
    ], dtype=np.float32)

    # Subtract the shift
    corrected_lab = gen_lab.copy()
    corrected_lab[:, :, 0] -= shift[0]
    corrected_lab[:, :, 1] -= shift[1]
    corrected_lab[:, :, 2] -= shift[2]

    return lab_to_rgb(corrected_lab), shift, corrected_lab


# ---------------------------------------------------------------------------
# Background subtraction primitives
# ---------------------------------------------------------------------------

def compute_difference_mask(
    generated: np.ndarray,
    background: np.ndarray,
    threshold: int = 10,
    ramp: int = 20,
    feather_radius: int = 4,
    diff_mode: str = 'lab',
    min_alpha: float = 0.0,
    morph_size: int = 5,
    guided_filter_eps: float = 0.02,
    *,
    gen_lab: Optional[np.ndarray] = None,
    bg_lab: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a soft difference mask between generated and background.

    Pipeline: difference → soft ramp → morphological cleanup → guided filter.

    Args:
        generated:  Generated frame float32 (HWC, 0-255).
        background: Background frame float32 (HWC, 0-255).
        threshold:  Low cutoff — differences below this are fully masked out.
        ramp:       Width of soft transition above threshold (higher = softer).
        feather_radius: Guided-filter radius in pixels (0 = skip).
        diff_mode:  ``'rgb'``, ``'lab'``, or ``'luminance'``.
        min_alpha:  Floor value for the mask (0.0-1.0).
        morph_size: Morphological-opening kernel size (0 = skip).
        guided_filter_eps: Guided-filter regularization.

    Returns:
        Mask as float32 (HW, 0.0-1.0).
    """
    # --- Per-pixel difference magnitude ---
    if diff_mode == 'rgb':
        diff = np.abs(generated - background)
        diff_magnitude = np.max(diff, axis=2)

    elif diff_mode == 'lab':
        if gen_lab is None:
            gen_lab = rgb_to_lab(generated)
        if bg_lab is None:
            bg_lab = rgb_to_lab(background)
        diff_magnitude = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))
        # Scale to ~0-255 so threshold values are consistent across modes
        diff_magnitude = np.clip(diff_magnitude * 2.55, 0, 255)

    elif diff_mode == 'luminance':
        gen_lum = (0.299 * generated[:, :, 0]
                   + 0.587 * generated[:, :, 1]
                   + 0.114 * generated[:, :, 2])
        bg_lum = (0.299 * background[:, :, 0]
                  + 0.587 * background[:, :, 1]
                  + 0.114 * background[:, :, 2])
        diff_magnitude = np.abs(gen_lum - bg_lum)

    else:
        raise ValueError(f"Unknown diff_mode: {diff_mode}")

    # --- Soft ramp threshold (replaces hard binary threshold) ---
    ramp_width = max(float(ramp), 1e-6)
    mask = np.clip(
        (diff_magnitude - threshold) / ramp_width, 0, 1
    ).astype(np.float32)

    # --- Morphological opening (remove small noise, then restore shapes) ---
    if morph_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        mask_u8 = (mask * 255).astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask = mask_u8.astype(np.float32) / 255.0

    # --- Edge-aware feathering via guided filter ---
    if feather_radius > 0:
        guide = cv2.cvtColor(
            generated.astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
        mask = guided_filter(guide, mask, feather_radius, guided_filter_eps)
        mask = np.clip(mask, 0, 1)

    # --- Apply minimum alpha ---
    if min_alpha > 0:
        mask = np.clip(mask, min_alpha, 1.0)

    return mask


def extract_creature(
    generated: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    gamma: float = 0.85,
) -> np.ndarray:
    """Extract the creature (bright additive elements) using the mask.

    Uses gamma correction to boost brightness relative to pixel intensity,
    producing smoother edges than a flat multiplier.

    Args:
        generated:  Generated frame float32 (HWC, 0-255).
        background: Background frame float32 (HWC, 0-255).
        mask:       Difference mask float32 (HW, 0.0-1.0).
        gamma:      Gamma correction (< 1.0 brightens, > 1.0 darkens).

    Returns:
        Extracted creature as uint8 (HWC, 0-255).
    """
    # Positive difference only (additive elements)
    creature_diff = np.clip(generated - background, 0, 255)

    # Gamma correction for smooth brightness boost
    normalized = creature_diff / 255.0
    boosted = np.power(normalized, gamma) * 255.0

    # Apply mask
    creature = boosted * mask[:, :, np.newaxis]

    return np.clip(creature, 0, 255).astype(np.uint8)
