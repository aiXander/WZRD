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

# Nano Banana 2.1's native output size per aspect at "1K" ("2K"/"4K" are
# exactly x2/x4). Every size measured from real outputs, 2026-10-10. It answers
# a reference of another aspect by cover-cropping it to the nearest of these,
# so a project canvas or region box at one of these ratios round-trips with
# no crop at all (D7).
NANO_BANANA_SIZES = {
    '1:1': (1024, 1024),
    '2:3': (848, 1264),
    '3:2': (1264, 848),
    '3:4': (896, 1200),
    '4:3': (1200, 896),
    '4:5': (928, 1152),
    '5:4': (1152, 928),
    '9:16': (768, 1376),
    '16:9': (1376, 768),
    '21:9': (1584, 672),
}

# Default base resolution (max dimension)
DEFAULT_BASE_RESOLUTION = 1920

# Aspect mismatch above which a generated image is reported as probably
# re-framed by the model (a warning, never a reason to stretch). Below it the
# mismatch is the model's own cover-crop to its native size — Nano Banana
# returns 1376x768 (1.792) for a 16:9 surface — which `uncrop_to_size` undoes
# exactly. Stretching instead was a constant 0.78 % vertical drift (D8).
ASPECT_RATIO_TOLERANCE = 0.02



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
    Center crop an image to exactly the target aspect ratio (a "cover" crop).

    Only a crop of less than one pixel is skipped; any real mismatch is
    cropped, never left for a later resize to stretch away.

    Args:
        image: PIL Image to crop
        target_aspect: Target aspect ratio (width/height)

    Returns:
        Cropped PIL Image
    """
    img_width, img_height = image.size
    current_aspect = img_width / img_height

    if current_aspect > target_aspect:
        # Image is wider than target: crop width
        new_width = int(round(img_height * target_aspect))
        new_height = img_height
    else:
        # Image is taller than target: crop height
        new_width = img_width
        new_height = int(round(img_width / target_aspect))

    if (new_width, new_height) == (img_width, img_height):
        return image  # Already the target aspect ratio

    # Calculate center crop coordinates
    left = (img_width - new_width) // 2
    top = (img_height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height))


def uncrop_to_size(
    image: Image.Image,
    size: Tuple[int, int],
    fill: Optional[Image.Image] = None,
) -> Tuple[Image.Image, dict]:
    """
    Undo a generator's cover-crop: place *image* back into a *size* frame.

    Image models answer a reference with an image at their own native size,
    cropped from the reference's centre ("cover"). The exact inverse is a
    uniform scale until the image fits inside *size* ("contain"), centred;
    the thin bands the model cropped away come back from *fill* (the
    reference itself, when the caller has it) or black. A uniform scale
    never distorts, so content lands where it was on the reference.

    Args:
        image: The generated (or otherwise model-cropped) image.
        size: Target (width, height) — the reference frame.
        fill: Optional image of *size* whose pixels fill the bands.

    Returns:
        Tuple of (image at *size*, info) — info has 'scale' and 'offset'
        (x, y of the pasted image, i.e. the band widths).
    """
    tw, th = size
    iw, ih = image.size
    scale = min(tw / iw, th / ih)
    nw, nh = min(tw, round(iw * scale)), min(th, round(ih * scale))
    resized = image if (nw, nh) == (iw, ih) else image.resize((nw, nh), Image.Resampling.LANCZOS)
    offset = ((tw - nw) // 2, (th - nh) // 2)
    info = {'scale': scale, 'offset': offset}
    if (nw, nh) == (tw, th):
        return resized, info
    if fill is not None and fill.size == (tw, th):
        canvas = fill.convert(resized.mode).copy()
    else:
        canvas = Image.new(resized.mode, (tw, th))
    canvas.paste(resized, offset)
    return canvas, info


def uncrop_array_to_size(
    array: np.ndarray,
    size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """`uncrop_to_size` for numpy arrays (masks, BGR frames): black bands."""
    tw, th = size
    ih, iw = array.shape[:2]
    scale = min(tw / iw, th / ih)
    nw, nh = min(tw, round(iw * scale)), min(th, round(ih * scale))
    resized = array if (nw, nh) == (iw, ih) else cv2.resize(array, (nw, nh), interpolation=interpolation)
    if (nw, nh) == (tw, th):
        return resized
    out = np.zeros((th, tw) + array.shape[2:], dtype=array.dtype)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def uncrop_box(
    box: Tuple[float, float, float, float],
    src_size: Tuple[int, int],
    size: Tuple[int, int],
) -> list:
    """Map an [x0, y0, x1, y1] box through the same transform as `uncrop_to_size`."""
    tw, th = size
    iw, ih = src_size
    scale = min(tw / iw, th / ih)
    nw, nh = min(tw, round(iw * scale)), min(th, round(ih * scale))
    sx, sy = nw / iw, nh / ih
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    x0, y0, x1, y1 = box
    return [round(x0 * sx + ox), round(y0 * sy + oy), round(x1 * sx + ox), round(y1 * sy + oy)]


def conform_to_size(
    image: Image.Image,
    size: Tuple[int, int],
) -> Image.Image:
    """Cover-crop an image to the aspect of *size*, then resize to exactly *size*.

    The import step for photos into a locked project canvas (D7): the photo
    loses its edges, never its proportions.
    """
    if image.size == tuple(size):
        return image
    cropped = center_crop_to_aspect(image, size[0] / size[1])
    return cropped.resize(tuple(size), Image.Resampling.LANCZOS)


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
    # Ensure even dimensions (required for video codecs)
    new_width -= new_width % 2
    new_height -= new_height % 2

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

    Same aspect → plain resize. Different aspect → the generator cropped the
    reference to its own native size, so that crop is undone exactly
    (`uncrop_to_size`, bands filled from the reference). Never stretches.

    Args:
        image: Image to align (e.g., AI-generated frame)
        reference: Reference image to match (e.g., darkened background)
        tolerance: Aspect mismatch above which a "probably re-framed" warning
            is added (the image is uncropped either way)

    Returns:
        Tuple of (aligned_image, info_dict)
        info_dict contains: 'resized', 'cropped' (always False), 'padded'
        (band offset x, y, or None), 'original_size', 'final_size', 'warnings'
    """
    ref_size = reference.size
    img_size = image.size

    info = {
        'resized': False,
        'cropped': False,
        'padded': None,
        'original_size': img_size,
        'final_size': ref_size,
        'warnings': []
    }

    # Check if already matching
    if img_size == ref_size:
        return image, info

    ref_aspect = get_aspect_ratio(ref_size)
    img_aspect = get_aspect_ratio(img_size)
    if not aspect_ratios_match(img_aspect, ref_aspect, tolerance):
        info['warnings'].append(
            f"Aspect ratio mismatch: image={img_aspect:.3f}, reference={ref_aspect:.3f} — "
            f"more than the model's usual native-size crop; it may have re-framed the scene."
        )

    result, uncrop = uncrop_to_size(image, ref_size, fill=reference)
    info['resized'] = True
    if uncrop['offset'] != (0, 0):
        info['padded'] = uncrop['offset']
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

    # Ensure even dimensions (required for video codecs)
    width -= width % 2
    height -= height % 2

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
    new_w = (w // factor) - ((w // factor) % 2)
    new_h = (h // factor) - ((h // factor) % 2)
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
# Resolution-relative size helpers
# ---------------------------------------------------------------------------

def frac_to_px(frac: float, ref_dim: int, *, odd: bool = False) -> int:
    """Convert a resolution-relative fraction to a pixel count.

    Args:
        frac: Fraction of *ref_dim* (e.g. 0.004 means 0.4 % of the
            reference dimension).
        ref_dim: Reference dimension in pixels (typically ``min(H, W)``).
        odd: If True, ensure the result is odd (useful for kernel sizes).

    Returns:
        Pixel count (at least 1 when *frac* > 0).
    """
    px = max(1, round(frac * ref_dim))
    if odd and px > 1 and px % 2 == 0:
        px += 1
    return px


# ---------------------------------------------------------------------------
# Background subtraction primitives
# ---------------------------------------------------------------------------

# Gaussian blur kernel size for smooth mask-modulated background subtraction
# (must be odd). Smoothing prevents high-frequency background texture from
# imprinting onto flat, bright creature regions.
_BG_SUBTRACT_BLUR_RADIUS = 31


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

    When ``subtract_bg`` is True, background subtraction strength is
    modulated by a heavily blurred (spatially smooth) inverse of the mask.
    In solid creature regions the mask is ~1 so almost no background is
    subtracted, avoiding high-frequency texture imprinting.  At the edges
    where the mask fades, full subtraction still removes the background
    cleanly.

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
        # Smooth the mask so subtraction strength varies spatially without
        # introducing high-frequency background texture into the creature.
        k = _BG_SUBTRACT_BLUR_RADIUS | 1  # ensure odd
        smooth_mask = cv2.GaussianBlur(mask, (k, k), 0)
        # subtraction_strength: 0 in creature core, 1 in background
        strength = (1.0 - smooth_mask)[:, :, np.newaxis]
        base = np.clip(generated - strength * background, 0, 255)
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
            w.write(frame_uint8)  # HWC uint8, rgb24
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 18,
        codec: str = 'libx264',
    ):
        self.output_path = Path(output_path)
        # Ensure even dimensions (required by most codecs / pixel formats)
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self._crop = (self.width != width or self.height != height)

        self._cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', codec,
            '-crf', str(crf),
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
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
