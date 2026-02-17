"""
Utility functions for projection mapping pipeline.

Handles resolution normalization, aspect ratio cropping, and image alignment
to ensure consistency across AI-generated content.
"""

from PIL import Image
import numpy as np
import subprocess
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



def _normalize_to_full_range(img: np.ndarray) -> np.ndarray:
    """Stretch a single-channel image to use the full 0-255 dynamic range."""
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _apply_colormap(gray: np.ndarray, colormap: int) -> np.ndarray:
    """Apply an OpenCV colormap to a grayscale image, ensuring full range.

    Returns a BGR uint8 image with full 0-255 dynamic range.
    """
    stretched = _normalize_to_full_range(gray)
    return cv2.applyColorMap(stretched, colormap)


def _lift_dark_pixels(frame_bgr: np.ndarray, base_color_bgr: Tuple[int, int, int] = (100, 30, 30)) -> np.ndarray:
    """Lift dark pixels towards a base color so they're visible on a projector.

    Uses per-channel ``np.maximum`` — dark pixels become the base color,
    bright pixels stay untouched.  Chosen default is a deep indigo that is
    clearly distinct from content.

    Args:
        frame_bgr: BGR uint8 image.
        base_color_bgr: Minimum per-channel floor (B, G, R).

    Returns:
        BGR uint8 image with no pixel darker than *base_color_bgr*.
    """
    floor = np.array(base_color_bgr, dtype=np.uint8).reshape(1, 1, 3)
    return np.maximum(frame_bgr, floor)


def _generate_alignment_aids(
    image: np.ndarray,
    *,
    output_dir: Optional[Union[str, Path]] = None,
    stem: str = "surface",
    video_fps: float = 24.0,
    hold_seconds: float = 3.0,
    crossfade_seconds: float = 1.5,
    base_color_bgr: Tuple[int, int, int] = (100, 30, 30),
) -> dict:
    """Generate a false-color alignment video for beamer calibration.

    Produces a seamlessly looping MP4 that slowly cycles through
    complementary colour visualisations.  Each aid is held for several
    seconds with gentle cross-fades, giving the operator time to study
    the projected image and make fine physical adjustments to the beamer.

    Because projection is additive (beamers can only ADD light), every
    frame is lifted so that no pixel is darker than *base_color_bgr*.
    This ensures that even featureless regions project enough light to
    reveal alignment errors.

    False-colour palettes exploit the human eye's superior sensitivity to
    colour differences — even tiny alignment errors become immediately
    visible as colour shifts.

    Args:
        image: Surface image (BGR or RGB uint8 HWC, 0-255).
        output_dir: Directory for the output MP4.
        stem: Filename prefix for the video.
        video_fps: Playback frame rate (higher = smoother fades).
        hold_seconds: Seconds each aid is shown before transitioning.
        crossfade_seconds: Seconds for each cross-fade transition.
        base_color_bgr: Minimum per-channel color floor (B, G, R).
            Dark pixels are lifted to this color so they remain visible
            on an additive projector.  Default deep indigo ``(100, 30, 30)``.

    Returns:
        Dict with ``'video_path'`` key (str) when *output_dir* is set,
        otherwise empty dict.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    aids: list = []  # list of BGR uint8 arrays

    # 1. Edges — directional Sobel mapped to HSV colour wheel
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 2.0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    angle = np.arctan2(sobel_y, sobel_x)  # -pi..pi
    hsv = np.zeros((*gray.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = _normalize_to_full_range(magnitude)
    aids.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    # 2. Adaptive texture — turbo colormap on CLAHE-enhanced luminance
    aids.append(_apply_colormap(enhanced, cv2.COLORMAP_TURBO))

    # 3. Difference of Gaussians — diverging hot/cold colormap
    sigma_lo, sigma_hi = 1.0, 6.0
    blur_lo = cv2.GaussianBlur(enhanced, (0, 0), sigma_lo)
    blur_hi = cv2.GaussianBlur(enhanced, (0, 0), sigma_hi)
    dog = blur_lo.astype(np.float32) - blur_hi.astype(np.float32)
    dog_norm = _normalize_to_full_range(dog)
    aids.append(_apply_colormap(dog_norm, cv2.COLORMAP_TWILIGHT_SHIFTED))

    # 4. Contours — luminance quantised into colour-coded elevation bands
    num_bands = 8
    quantised = np.clip(
        (gray.astype(np.float32) / 255.0 * num_bands).astype(np.int32),
        0, num_bands - 1,
    )
    band_hue = (quantised.astype(np.float32) / num_bands * 179).astype(np.uint8)
    band_val = _normalize_to_full_range(gray)
    hsv_bands = np.zeros((*gray.shape, 3), dtype=np.uint8)
    hsv_bands[:, :, 0] = band_hue
    hsv_bands[:, :, 1] = 200
    hsv_bands[:, :, 2] = np.clip(band_val.astype(np.int32) + 80, 0, 255).astype(np.uint8)
    contour_bgr = cv2.cvtColor(hsv_bands, cv2.COLOR_HSV2BGR)
    for band in range(1, num_bands):
        boundary = (quantised == band).astype(np.uint8) * 255
        edges_band = cv2.Canny(boundary, 50, 150)
        edges_band = cv2.dilate(edges_band, np.ones((2, 2), np.uint8), iterations=1)
        contour_bgr[edges_band > 0] = 255
    aids.append(contour_bgr)

    # Lift dark pixels on ALL aids so they're visible on projector
    aids = [_lift_dark_pixels(a, base_color_bgr) for a in aids]

    # Generate seamlessly looping video
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Dimmed original as a rest frame between aids
        dimmed_gray = _normalize_to_full_range((gray * 0.3).astype(np.uint8))
        dimmed_color = _apply_colormap(dimmed_gray, cv2.COLORMAP_BONE)
        dimmed_color = _lift_dark_pixels(dimmed_color, base_color_bgr)

        # Convert all key frames to RGB for the video writer
        key_frames_bgr = [dimmed_color] + aids
        key_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in key_frames_bgr]
        # Pre-compute float32 versions for crossfade blending
        key_frames_f = [f.astype(np.float32) for f in key_frames]

        h, w = gray.shape[:2]
        video_path = out / f"{stem}_alignment_cycle.mp4"

        # Frame counts from real-time durations
        hold_frames = max(int(video_fps * hold_seconds), 1)
        crossfade_frames = max(int(video_fps * crossfade_seconds), 1)

        # Stream frames directly to ffmpeg (no intermediate list).
        # Hold-frame bytes are serialised once and written N times.
        n = len(key_frames)
        # Pre-crop hold frames so tobytes matches what ffmpeg expects
        crop_h = h - (h % 2)
        crop_w = w - (w % 2)
        key_raw = [f[:crop_h, :crop_w].tobytes() for f in key_frames]

        with VideoWriter(video_path, w, h, video_fps) as writer:
            for i in range(n):
                src_f = key_frames_f[i]
                dst_f = key_frames_f[(i + 1) % n]
                # Hold — write pre-serialised bytes
                writer.write_raw(key_raw[i], hold_frames)
                # Cross-fade
                for t in range(crossfade_frames):
                    alpha = (t + 1) / (crossfade_frames + 1)
                    blended = ((1 - alpha) * src_f + alpha * dst_f).astype(np.uint8)
                    writer.write(blended)

        return {'video_path': str(video_path)}

    return {}


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
# Multi-resolution helpers
# ---------------------------------------------------------------------------

def downscale_arr(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downscale array by integer factor using area interpolation.

    Args:
        arr: Input array (HW or HWC, any dtype).
        factor: Integer downscale factor (2 = half, 4 = quarter).

    Returns:
        Downscaled array.
    """
    h, w = arr.shape[:2]
    return cv2.resize(arr, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def upscale_arr(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Upscale array to target (height, width) using bilinear interpolation.

    Args:
        arr: Input array (HW or HWC, any dtype).
        target_hw: Target (height, width).

    Returns:
        Upscaled array.
    """
    return cv2.resize(arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)


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


def compute_color_shift(
    gen_lab: np.ndarray,
    bg_lab: np.ndarray,
    percentile: float = 50,
) -> np.ndarray:
    """Compute global LAB color shift between generated and background.

    Estimates the systematic color offset using confident background pixels
    (those with the smallest per-pixel LAB difference).  Designed to run on
    downscaled inputs for speed — only the 3-element shift vector is needed.

    Args:
        gen_lab: Generated frame in LAB (HWC, float32).
        bg_lab: Background frame in LAB (HWC, float32).
        percentile: Use lowest N% of difference pixels as reference.

    Returns:
        3-element float32 shift vector (L, a, b).
    """
    diff = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))
    cutoff = np.percentile(diff, percentile)
    bg_mask = diff <= cutoff

    if np.sum(bg_mask) < 100:
        return np.zeros(3, dtype=np.float32)

    return np.array([
        np.mean(gen_lab[bg_mask, 0] - bg_lab[bg_mask, 0]),
        np.mean(gen_lab[bg_mask, 1] - bg_lab[bg_mask, 1]),
        np.mean(gen_lab[bg_mask, 2] - bg_lab[bg_mask, 2]),
    ], dtype=np.float32)


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


# Toggle between smooth mask-modulated background subtraction (True) and
# legacy per-pixel subtraction (False).  The smooth mode prevents
# high-frequency background texture from imprinting onto flat, bright
# creature regions.  Flip to False to restore the original behavior.
SMOOTH_BG_SUBTRACT = False
SMOOTH_BG_SUBTRACT_RADIUS = 31   # Gaussian blur kernel size (must be odd)


def extract_creature(
    generated: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    gamma: float = 0.85,
    subtract_bg: bool = False,
) -> np.ndarray:
    """Extract the creature (bright additive elements) using the mask.

    Uses gamma correction to boost brightness relative to pixel intensity,
    producing smoother edges than a flat multiplier.

    When ``subtract_bg`` is True and ``SMOOTH_BG_SUBTRACT`` is enabled,
    background subtraction strength is modulated by a heavily blurred
    (spatially smooth) inverse of the mask.  In solid creature regions
    the mask is ~1 so almost no background is subtracted, avoiding
    high-frequency texture imprinting.  At the edges where the mask
    fades, full subtraction still removes the background cleanly.

    Args:
        generated:  Generated frame float32 (HWC, 0-255).
        background: Background frame float32 (HWC, 0-255).
        mask:       Difference mask float32 (HW, 0.0-1.0).
        gamma:      Gamma correction (< 1.0 brightens, > 1.0 darkens).
        subtract_bg: If True, subtract background before masking.
            If False, use the generated frame directly with the mask as alpha.

    Returns:
        Extracted creature as uint8 (HWC, 0-255).
    """
    if subtract_bg:
        if SMOOTH_BG_SUBTRACT:
            # Smooth the mask so subtraction strength varies spatially without
            # introducing high-frequency background texture into the creature.
            k = SMOOTH_BG_SUBTRACT_RADIUS | 1  # ensure odd
            smooth_mask = cv2.GaussianBlur(mask, (k, k), 0)
            # subtraction_strength: 0 in creature core, 1 in background
            strength = (1.0 - smooth_mask)[:, :, np.newaxis]
            base = np.clip(generated - strength * background, 0, 255)
        else:
            # Legacy per-pixel subtraction
            base = np.clip(generated - background, 0, 255)
    else:
        # Mask-only: keep generated frame colors, let the mask handle fading
        base = generated

    # Gamma correction for smooth brightness boost
    normalized = base / 255.0
    boosted = np.power(normalized, gamma) * 255.0

    # Apply mask
    creature = boosted * mask[:, :, np.newaxis]

    return np.clip(creature, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# FFmpeg video writer
# ---------------------------------------------------------------------------

class VideoWriter:
    """Context-managed FFmpeg video writer.

    Streams raw frames to an FFmpeg subprocess that encodes to H.264/mp4
    with settings optimised for broad device compatibility.

    Usage::

        with VideoWriter(path, width, height, fps) as w:
            w.write(frame_uint8)  # HWC uint8, rgb24 or rgba
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: float,
        *,
        alpha: bool = False,
        crf: int = 18,
        codec: str = 'libx264',
    ):
        self.output_path = Path(output_path)
        # Ensure even dimensions (required by most codecs / pixel formats)
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self._crop = (self.width != width or self.height != height)

        if alpha:
            input_pix_fmt = 'rgba'
            output_pix_fmt = 'yuva444p10le'
            output_codec = 'prores_ks'
            extra_args = ['-profile:v', '4444']
        else:
            input_pix_fmt = 'rgb24'
            output_pix_fmt = 'yuv420p'
            output_codec = codec
            extra_args = ['-crf', str(crf), '-movflags', '+faststart']

        self._cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', input_pix_fmt,
            '-r', str(fps),
            '-i', '-',
            '-c:v', output_codec,
            *extra_args,
            '-pix_fmt', output_pix_fmt,
            '-v', 'error',
            str(self.output_path),
        ]
        self._process: Optional[subprocess.Popen] = None

    def __enter__(self):
        self._process = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame (uint8 HWC array)."""
        if self._crop:
            frame = frame[:self.height, :self.width]
        self._process.stdin.write(frame.tobytes())

    def write_raw(self, raw: bytes, count: int = 1) -> None:
        """Write pre-serialised frame bytes *count* times."""
        for _ in range(count):
            self._process.stdin.write(raw)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._process.stdin.close()
        self._process.wait()
        return False
