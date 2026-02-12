"""
Projection surface detection for projection mapping.

Detects the brightest quadrilateral region in an image (typically a projector's
light surface) and returns a perspective-corrected crop along with corner coordinates.

The algorithm:
1. Compute color proximity map highlighting the projection
2. Apply Otsu's thresholding (automatic threshold for bimodal histogram)
3. Morphological closing to fill holes and smooth edges
4. Find contours and select the largest one
5. Compute convex hull and minimum area bounding rectangle to get 4 corners

Example usage:
    from wzrd import detect_projection_area, detect_projection_from_array

    # Detect from file and save outputs
    cropped, info = detect_projection_area(
        "photo.jpg",
        margin=0.02,  # 2% smaller than detected region
        output_path="cropped.png",
        preview_path="preview.png"
    )
    corners = info['corners']  # Normalized coordinates

    # Detect from numpy array
    cropped, info = detect_projection_from_array(image_array, margin=0.02)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Union
from pathlib import Path
import os

from .utils import parse_aspect_ratio, compute_target_dimensions, DebugContext

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in a consistent manner: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of 4 points (4, 2)

    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: top-left has smallest, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Difference of coordinates: top-right has smallest, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def compute_luminance(image: np.ndarray) -> np.ndarray:
    """
    Compute perceptual luminance from BGR image.
    Uses weighted sum that matches human perception.
    """
    # Convert to float for precision
    img_float = image.astype(np.float32)

    # Perceptual luminance weights (ITU-R BT.709)
    luminance = 0.0722 * img_float[:, :, 0] + 0.7152 * img_float[:, :, 1] + 0.2126 * img_float[:, :, 2]

    return luminance


def compute_saturation_boosted_brightness(
    image: np.ndarray,
    saturation_floor: float = 0.3,
    saturation_boost: float = 2.0,
) -> np.ndarray:
    """
    Compute brightness that also considers high saturation (colored projections like blue).
    Projections must have BOTH high saturation AND high brightness to score well.

    Designed for detecting projector light (blue, white, red, etc.) against outdoor
    organic scenes where ambient objects have moderate saturation but aren't as bright.

    Args:
        image: BGR input image
        saturation_floor: Minimum saturation (0-1) to contribute to score.
                          Below this, only luminance matters. Default 0.3 suppresses
                          low-saturation ambient light (foliage, soil, etc.)
        saturation_boost: Multiplier for the saturation*value term. Default 2.0
                          ensures fully saturated bright pixels score 1.0.

    Returns:
        Brightness map as uint8 image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    # Also compute luminance
    luminance = compute_luminance(image)

    # Normalize to 0-1 range
    v_norm = v / 255.0
    s_norm = s / 255.0
    lum_norm = luminance / 255.0

    # Saturation floor: ignore low saturation (ambient light, desaturated colors)
    # Soft ramp from 0 at saturation_floor to 1.0 at full saturation
    s_effective = np.clip((s_norm - saturation_floor) / (1.0 - saturation_floor), 0, 1)

    # Projection indicator: requires BOTH high saturation AND high brightness
    # The v_norm² term emphasizes that brightness must be high - a saturated but
    # dim pixel won't score well. This separates projector light from ambient.
    #
    # Examples with saturation_floor=0.3, saturation_boost=2.0:
    #   Blue projection (S=1.0, V=1.0): s_eff=1.0, score = 1.0 * 1.0 * 2.0 = 2.0 → 1.0
    #   Red projection (S=1.0, V=1.0):  same → 1.0
    #   White projection: luminance=1.0 handles it
    #   Foliage (S=0.4, V=0.6): s_eff=0.14, score = 0.14 * 0.36 * 2.0 = 0.10
    #   Dark saturated (S=0.8, V=0.4): s_eff=0.71, score = 0.71 * 0.16 * 2.0 = 0.23
    saturation_term = s_effective * v_norm * v_norm * saturation_boost

    projection_score = np.maximum(lum_norm, saturation_term)
    projection_score = np.clip(projection_score, 0, 1)

    return (projection_score * 255).astype(np.uint8)


def extract_dominant_color_from_brightness(
    image: np.ndarray,
    brightness: np.ndarray,
    debug: Optional['DebugContext'] = None
) -> Optional[Tuple[float, float, float]]:
    """
    Extract the dominant projection color weighted by brightness.

    Uses the brightness map as weights to compute the weighted average color.
    This avoids needing a binary mask - the blur naturally highlights the projection.

    Args:
        image: BGR input image
        brightness: Grayscale brightness map (0-255), typically blurred
        debug: Optional debug context for visualization

    Returns:
        (H, S, V) tuple of the dominant color, or None if extraction fails.
        H is in [0, 180], S and V are in [0, 255].
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize brightness to use as weights
    weights = brightness.astype(np.float64) / 255.0

    # Only consider reasonably bright pixels (top ~50% of brightness range)
    threshold = np.percentile(weights, 50)
    mask = weights > threshold
    if not np.any(mask):
        return None

    weights_masked = weights[mask]
    h_pixels = hsv[:, :, 0][mask].astype(np.float64)
    s_pixels = hsv[:, :, 1][mask].astype(np.float64)
    v_pixels = hsv[:, :, 2][mask].astype(np.float64)

    # Weighted circular mean for hue (handles wraparound)
    h_rad = h_pixels * np.pi / 90.0  # H is 0-180, convert to radians
    sin_h = np.sum(np.sin(h_rad) * weights_masked)
    cos_h = np.sum(np.cos(h_rad) * weights_masked)
    h_mean = (np.arctan2(sin_h, cos_h) * 90.0 / np.pi) % 180

    # Weighted mean for S and V
    total_weight = np.sum(weights_masked)
    s_mean = np.sum(s_pixels * weights_masked) / total_weight
    v_mean = np.sum(v_pixels * weights_masked) / total_weight

    dominant_color = (float(h_mean), float(s_mean), float(v_mean))

    if debug and debug.enabled:
        # Create a visualization of the detected color
        color_vis = np.zeros((100, 200, 3), dtype=np.uint8)
        color_vis[:, :] = [int(h_mean), int(s_mean), int(v_mean)]
        color_vis_bgr = cv2.cvtColor(color_vis, cv2.COLOR_HSV2BGR)
        cv2.putText(color_vis_bgr, f"H:{h_mean:.1f} S:{s_mean:.1f} V:{v_mean:.1f}",
                    (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        debug.save("detected_projection_color", color_vis_bgr)
        print(f"  [debug] Detected projection color: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")

    return dominant_color


def compute_color_proximity(
    image: np.ndarray,
    target_color_hsv: Tuple[float, float, float],
    decay: float = 3.5,
    weight_v: float = 0.5,
    weight_h: float = 0.3,
    weight_s: float = 0.2,
    brightness_base: float = 0.23,
    brightness_scale: float = 0.77,
) -> np.ndarray:
    """
    Compute how close each pixel is to the target projection color.

    Returns a grayscale image where brighter = closer to target color.
    This replaces luminance detection with color-proximity detection.

    Uses a soft combination that considers:
    - Color similarity (hue + saturation)
    - Brightness similarity (value)
    - Overall brightness boost (projections are always bright)

    Output is rescaled to full dynamic range (min=0, max=255).

    Args:
        image: BGR input image
        target_color_hsv: (H, S, V) target color. H in [0,180], S,V in [0,255]
        decay: Exponential decay factor (higher = tighter color selection)
        weight_v: Weight for value/brightness distance
        weight_h: Weight for hue distance
        weight_s: Weight for saturation distance
        brightness_base: Base brightness factor (0-1)
        brightness_scale: Scale for brightness factor (brightness_base + brightness_scale * v)

    Returns:
        Grayscale proximity map (uint8, 0-255) with full dynamic range
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    target_h, target_s, target_v = target_color_hsv

    # Compute hue distance (circular, 0-90 range since max diff is 90 in 0-180 space)
    h_diff = np.minimum(np.abs(h - target_h), 180 - np.abs(h - target_h))
    h_dist_norm = h_diff / 90.0  # Normalize to 0-1

    # Compute saturation and value distances (normalized to 0-1)
    s_dist_norm = np.abs(s - target_s) / 255.0
    v_dist_norm = np.abs(v - target_v) / 255.0

    # For low saturation targets (near white/gray), hue is unreliable
    if target_s < 50:
        # Weight: mostly value, some saturation, ignore hue
        color_dist = 0.7 * v_dist_norm + 0.3 * s_dist_norm
    else:
        # Weight: configurable
        color_dist = weight_v * v_dist_norm + weight_h * h_dist_norm + weight_s * s_dist_norm

    # Convert distance to similarity (0 = far, 1 = close)
    # Use a soft falloff instead of hard threshold
    proximity = np.exp(-decay * color_dist)

    # Boost pixels that are also bright (projections are always brighter than ambient)
    brightness_factor = v / 255.0
    proximity = proximity * (brightness_base + brightness_scale * brightness_factor)

    # Rescale to full dynamic range: min=0, max=1
    p_min = proximity.min()
    p_max = proximity.max()
    if p_max > p_min:
        proximity = (proximity - p_min) / (p_max - p_min)
    else:
        proximity = np.zeros_like(proximity)

    return (proximity * 255).astype(np.uint8)


def resize_for_optimization(image: np.ndarray, max_size: int = 512) -> Tuple[np.ndarray, float]:
    """
    Resize image for optimization, preserving aspect ratio.
    Returns resized image and scale factor.
    """
    h, w = image.shape[:2]
    scale = min(max_size / max(h, w), 1.0)  # Don't upscale
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    return image, 1.0


def find_quadrilateral_from_proximity(
    proximity: np.ndarray,
    min_projection_fraction: float = 0.15,
    debug: Optional['DebugContext'] = None,
    verbose: bool = True
) -> Optional[np.ndarray]:
    """
    Find a 4-corner quadrilateral from a proximity/brightness image.

    Uses a simple and robust approach:
    1. Otsu's thresholding to separate bright foreground from dark background
    2. Morphological closing to fill holes and smooth edges
    3. Find contours and select the largest one
    4. Convex hull + minimum area bounding rectangle to get a quadrilateral
       that fully contains the entire mask

    Args:
        proximity: Grayscale proximity/brightness image (0-255)
        min_projection_fraction: Minimum fraction of image area the polygon must cover
        debug: Optional debug context for saving intermediate images
        verbose: Whether to print progress messages

    Returns:
        4 corners as (4, 2) array in order: TL, TR, BR, BL, or None if detection fails
    """
    h, w = proximity.shape
    image_area = h * w

    # Step 1: Otsu's thresholding
    # Automatically calculates optimal threshold for bimodal histogram
    threshold_value, binary = cv2.threshold(
        proximity, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if verbose:
        print(f"  [otsu] Automatic threshold: {threshold_value:.1f}")

    if debug and debug.enabled:
        debug.save("otsu_threshold", binary)

    # Step 2: Morphological closing to fill holes and smooth edges
    # Kernel size proportional to image size
    kernel_size = max(5, int(min(h, w) * 0.02))
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    if verbose:
        print(f"  [morph] Applied closing with {kernel_size}x{kernel_size} kernel")

    if debug and debug.enabled:
        debug.save("morphological_closed", closed)

    # Step 3: Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if verbose:
            print("  [contour] No contours found")
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)

    if verbose:
        print(f"  [contour] Largest contour area: {contour_area / image_area * 100:.1f}% of image")

    # Check minimum area constraint
    if contour_area < image_area * min_projection_fraction:
        if verbose:
            print(f"  [contour] Contour too small (< {min_projection_fraction * 100:.0f}% of image)")
        return None

    # Visualize the largest contour
    if debug and debug.enabled:
        contour_vis = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_vis, [largest_contour], -1, (0, 255, 0), 2)
        debug.save("largest_contour", contour_vis)

    # Step 4: Find minimum area bounding rectangle that contains the entire mask
    # Use convex hull + minAreaRect to get a quadrilateral that fully encompasses
    # all mask pixels (unlike Douglas-Peucker which approximates the contour shape
    # and may cut off parts of the mask)
    hull = cv2.convexHull(largest_contour)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    corners = box.astype(np.float32)

    if verbose:
        print(f"  [rect] Found bounding rectangle via minAreaRect on convex hull")

    # Order the corners consistently: TL, TR, BR, BL
    corners = order_points(corners)

    # Clamp to image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    # Visualize the final polygon
    if debug and debug.enabled:
        poly_vis = cv2.cvtColor(proximity, cv2.COLOR_GRAY2BGR)
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(poly_vis, [pts], True, (0, 255, 0), 2)
        for i, corner in enumerate(corners):
            pt = (int(corner[0]), int(corner[1]))
            cv2.circle(poly_vis, pt, 5, (0, 0, 255), -1)
            labels = ['TL', 'TR', 'BR', 'BL']
            cv2.putText(poly_vis, labels[i], (pt[0] + 8, pt[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        debug.save("detected_quadrilateral", poly_vis)

    return corners


def detect_projection_optimized(
    image: np.ndarray,
    blur_factor: float = 0.08,
    optimization_size: int = 512,
    debug: Optional[DebugContext] = None,
    use_color_refinement: bool = True,
    min_projection_fraction: float = 0.15
) -> Optional[np.ndarray]:
    """
    Detect projection surface using Otsu thresholding and contour analysis.

    Approach:
    1. Downscale image for faster processing
    2. Compute combined brightness (luminance + saturation boost)
    3. Apply blur to create smooth gradients
    4. Extract dominant projection color
    5. Compute color proximity map
    6. Apply Otsu's thresholding for automatic binarization
    7. Morphological closing to fill holes
    8. Find largest contour and approximate to 4-point polygon

    Args:
        image: BGR input image
        blur_factor: Blur kernel size as fraction of image dimension
        optimization_size: Max dimension for processing (preserves aspect ratio)
        debug: Debug context for saving intermediate images
        use_color_refinement: If True, use color-proximity detection (recommended)
        min_projection_fraction: Minimum fraction of image area the polygon must cover

    Returns:
        4 corner points (TL, TR, BR, BL) or None if detection fails
    """
    h_orig, w_orig = image.shape[:2]
    if debug is None:
        debug = DebugContext(None)

    # Step 1: Downscale for initial color detection
    image_small, scale = resize_for_optimization(image, optimization_size)
    h, w = image_small.shape[:2]
    debug.save("input_downscaled", image_small)

    # Step 2: Compute combined brightness (luminance + saturation boost)
    luminance = compute_luminance(image_small)
    debug.save("luminance_raw", luminance, normalize=True)

    sat_bright = compute_saturation_boosted_brightness(image_small)
    debug.save("saturation_boosted_brightness", sat_bright)

    combined = np.maximum(luminance, sat_bright.astype(np.float32))
    debug.save("combined_brightness", combined, normalize=True)

    # Step 3: Apply blur to create smooth gradients highlighting projection
    blur_size = int(max(h, w) * blur_factor)
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    blur_size = max(blur_size, 21)

    blurred = cv2.GaussianBlur(combined, (blur_size, blur_size), 0)
    blurred_norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    debug.save("blurred_for_color_extraction", blurred_norm)

    # Step 4: Extract dominant projection color and compute proximity map
    proximity_for_optim = None
    if use_color_refinement:
        dominant_color = extract_dominant_color_from_brightness(image_small, blurred_norm, debug)

        if dominant_color is not None:
            if debug.enabled:
                print(f"  [detect] Computing color proximity...")

            # Compute color proximity on the downscaled image for speed
            proximity = compute_color_proximity(image_small, dominant_color)
            debug.save("color_proximity_raw", proximity)

            # Apply blur to smooth out projection content details
            # and create a clean bright blob for the projection region
            heavy_blur_size = int(max(h, w) * 0.08)  # Moderate blur
            heavy_blur_size = heavy_blur_size if heavy_blur_size % 2 == 1 else heavy_blur_size + 1
            heavy_blur_size = max(heavy_blur_size, 15)

            proximity_blurred = cv2.GaussianBlur(proximity, (heavy_blur_size, heavy_blur_size), 0)
            debug.save("color_proximity_blurred", proximity_blurred)
            proximity_for_optim = proximity_blurred
        else:
            if debug.enabled:
                print(f"  [detect] Color extraction failed, using brightness only...")
            use_color_refinement = False

    if not use_color_refinement or proximity_for_optim is None:
        # Fallback: use blurred brightness directly
        if debug.enabled:
            print(f"  [detect] Using brightness for polygon detection...")
        proximity_for_optim = blurred_norm

    debug.save("proximity_for_detection", proximity_for_optim)

    # Step 5: Find quadrilateral using Otsu + morphology + contours + Douglas-Peucker
    if debug.enabled:
        print(f"\n  [detect] === Finding quadrilateral ===")

    corners = find_quadrilateral_from_proximity(
        proximity_for_optim,
        min_projection_fraction=min_projection_fraction,
        debug=debug,
        verbose=debug.enabled
    )

    if corners is None:
        if debug.enabled:
            print(f"  [detect] Quadrilateral detection failed")
        return None

    # Scale corners back to full resolution
    corners = corners / scale
    corners = order_points(corners)

    # Clamp to original image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, w_orig - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h_orig - 1)

    debug.save_with_corners("detected_corners", image, corners, (0, 255, 0))

    return corners


def iterative_detection(
    image: np.ndarray,
    blur_factors: List[float] = [0.08, 0.06, 0.10, 0.12],
    debug: Optional[DebugContext] = None,
    use_color_refinement: bool = True,
    min_projection_fraction: float = 0.15
) -> Optional[np.ndarray]:
    """
    Detect projection quadrilateral with fallback blur factors.

    Tries the primary detection first, then falls back to different blur
    factors if needed.

    Args:
        image: BGR input image
        blur_factors: List of blur factors to try if primary fails
        debug: Optional debug context
        use_color_refinement: If True, use color-proximity detection
        min_projection_fraction: Minimum fraction of image area the polygon must cover

    Returns:
        4 corners (TL, TR, BR, BL) or None if detection fails
    """
    h, w = image.shape[:2]
    if debug is None:
        debug = DebugContext(None)

    def is_valid_detection(corners):
        """Check if detection is reasonable (not just the whole image)."""
        margin = 0.05
        corners_norm = corners / np.array([w, h])
        edge_count = np.sum(
            (corners_norm[:, 0] < margin) |
            (corners_norm[:, 0] > 1 - margin) |
            (corners_norm[:, 1] < margin) |
            (corners_norm[:, 1] > 1 - margin)
        )
        # Also check area isn't too small or too large
        area = cv2.contourArea(corners)
        area_ratio = area / (h * w)
        return edge_count <= 2 and min_projection_fraction < area_ratio < 0.98

    # Try primary detection with default blur and color refinement
    corners = detect_projection_optimized(
        image, debug=debug, use_color_refinement=use_color_refinement,
        min_projection_fraction=min_projection_fraction
    )
    if corners is not None and is_valid_detection(corners):
        debug.save_with_corners("detection_success", image, corners, (0, 255, 0))
        return corners

    if debug.enabled:
        print("  [detect] Primary detection failed, trying different blur factors...")

    # Try different blur factors
    for blur_factor in blur_factors:
        corners = detect_projection_optimized(
            image,
            blur_factor=blur_factor,
            debug=None,  # Don't re-save debug images
            use_color_refinement=use_color_refinement,
            min_projection_fraction=min_projection_fraction
        )
        if corners is not None and is_valid_detection(corners):
            debug.save_with_corners(f"detection_blur{blur_factor:.2f}", image, corners, (0, 255, 0))
            return corners

    # If color refinement failed, try without it as fallback
    if use_color_refinement:
        if debug.enabled:
            print("  [detect] Color refinement failed, trying without...")
        corners = detect_projection_optimized(
            image, blur_factor=0.08, debug=None, use_color_refinement=False,
            min_projection_fraction=min_projection_fraction
        )
        if corners is not None and is_valid_detection(corners):
            debug.save_with_corners("detection_no_color_refinement", image, corners, (0, 255, 0))
            return corners

    # Return best effort without validation
    corners = detect_projection_optimized(
        image, blur_factor=0.08, debug=None, use_color_refinement=False,
        min_projection_fraction=min_projection_fraction
    )
    if corners is not None:
        debug.save_with_corners("detection_best_effort", image, corners, (128, 128, 255))
    return corners


def apply_margin_to_corners(
    corners: np.ndarray,
    margin: float,
    image_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Adjust corners by a margin percentage.

    Args:
        corners: Array of 4 points (4, 2), ordered TL, TR, BR, BL
        margin: Fraction to adjust (positive = shrink inward, negative = expand outward)
                e.g., 0.02 = 2% smaller, -0.02 = 2% larger
        image_size: Optional (width, height) to clamp corners within bounds

    Returns:
        Adjusted corners array, clamped to image bounds if image_size provided
    """
    if margin == 0:
        adjusted = corners.copy()
    else:
        # Compute centroid
        centroid = corners.mean(axis=0)

        # Move each corner toward/away from centroid by margin percentage
        # Positive margin moves corners toward centroid (shrinks)
        # Negative margin moves corners away from centroid (expands)
        adjusted = corners.copy().astype(np.float32)
        for i in range(4):
            direction = centroid - corners[i]
            adjusted[i] = corners[i] + direction * margin

    # Clamp to image bounds if provided
    if image_size is not None:
        w, h = image_size
        adjusted[:, 0] = np.clip(adjusted[:, 0], 0, w - 1)
        adjusted[:, 1] = np.clip(adjusted[:, 1], 0, h - 1)

    return adjusted


def compute_natural_dimensions(corners: np.ndarray) -> Tuple[float, float]:
    """
    Compute the natural output dimensions from a quadrilateral.

    Args:
        corners: 4 corner points (TL, TR, BR, BL)

    Returns:
        (width, height) as floats
    """
    width_top = np.linalg.norm(corners[1] - corners[0])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    height_left = np.linalg.norm(corners[3] - corners[0])
    height_right = np.linalg.norm(corners[2] - corners[1])

    width = (width_top + width_bottom) / 2
    height = (height_left + height_right) / 2

    return width, height


def adjust_corners_for_aspect_ratio(
    corners: np.ndarray,
    target_aspect: float,
    tolerance: float = 0.01
) -> np.ndarray:
    """
    Adjust source corners to achieve target aspect ratio in warped output.

    This selects a subset of the quadrilateral by interpolating along edges,
    so that when the adjusted quad is perspective-warped, the result has
    exactly the target aspect ratio (center-cropped from the original region).

    Args:
        corners: 4 corner points (TL, TR, BR, BL) as (4, 2) array
        target_aspect: Target aspect ratio (width/height)
        tolerance: Skip adjustment if natural aspect is within this tolerance

    Returns:
        Adjusted corners that will produce target aspect ratio when warped
    """
    corners = corners.astype(np.float32)
    natural_width, natural_height = compute_natural_dimensions(corners)
    natural_aspect = natural_width / natural_height

    if abs(natural_aspect - target_aspect) / target_aspect < tolerance:
        return corners.copy()

    adjusted = corners.copy()

    if natural_aspect > target_aspect:
        # Natural is wider than target: crop width (inset left and right edges)
        new_width = natural_height * target_aspect
        crop_fraction = (natural_width - new_width) / (2 * natural_width)

        # Move TL and BL inward along top/bottom edges
        adjusted[0] = corners[0] + crop_fraction * (corners[1] - corners[0])  # TL toward TR
        adjusted[3] = corners[3] + crop_fraction * (corners[2] - corners[3])  # BL toward BR

        # Move TR and BR inward along top/bottom edges
        adjusted[1] = corners[1] - crop_fraction * (corners[1] - corners[0])  # TR toward TL
        adjusted[2] = corners[2] - crop_fraction * (corners[2] - corners[3])  # BR toward BL
    else:
        # Natural is taller than target: crop height (inset top and bottom edges)
        new_height = natural_width / target_aspect
        crop_fraction = (natural_height - new_height) / (2 * natural_height)

        # Move TL and TR downward along left/right edges
        adjusted[0] = corners[0] + crop_fraction * (corners[3] - corners[0])  # TL toward BL
        adjusted[1] = corners[1] + crop_fraction * (corners[2] - corners[1])  # TR toward BR

        # Move BL and BR upward along left/right edges
        adjusted[3] = corners[3] - crop_fraction * (corners[3] - corners[0])  # BL toward TL
        adjusted[2] = corners[2] - crop_fraction * (corners[2] - corners[1])  # BR toward TR

    return adjusted


def warp_perspective_crop(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extract and rectify the quadrilateral region from the image.

    Args:
        image: Source BGR image
        corners: 4 corner points (TL, TR, BR, BL) in pixel coordinates
        output_size: Optional (width, height) for output. If None, computed from corners.

    Returns:
        Perspective-corrected rectangular crop
    """
    corners = corners.astype(np.float32)

    # Compute output dimensions from the quadrilateral if not specified
    if output_size is None:
        # Width: average of top and bottom edge lengths
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))

        # Height: average of left and right edge lengths
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = int(max(height_left, height_right))
    else:
        width, height = output_size

    # Ensure minimum size
    width = max(width, 10)
    height = max(height, 10)

    # Define destination rectangle corners
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dst_corners)

    # Warp the image
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped


def detect_projection_from_array(
    image: np.ndarray,
    margin: float = 0.02,
    debug_dir: Optional[str] = None,
    min_projection_fraction: float = 0.10,
    target_aspect_ratio: Optional[str] = None,
    output_resolution: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Detect the projection area from a numpy array and return cropped image.

    Args:
        image: BGR input image as numpy array
        margin: Fraction to shrink crop inward (0.02 = 2% smaller than detected region)
        debug_dir: Optional directory to save debug images. If provided, intermediate
                   steps will be saved with numbered prefixes (01_, 02_, etc.)
        min_projection_fraction: Minimum fraction of image area the projection must cover
                                 (default: 0.10 = 10% of image)
        target_aspect_ratio: Optional target aspect ratio for output (e.g., "16:9", "4:3").
                             If provided, the source region will be center-cropped to achieve
                             exactly this aspect ratio in the output.
        output_resolution: Optional max dimension for output. If provided along with
                           target_aspect_ratio, output will be resized to exact dimensions.

    Returns:
        Tuple of (cropped_image, info_dict):
            - cropped_image: Perspective-corrected crop of the projection area (BGR uint8)
            - info_dict containing:
                - corners: List of 4 (x, y) tuples normalized to [0, 1] (TL, TR, BR, BL)
                - corners_pixel: Corner coordinates in pixels
                - preview: Preview image with detected polygon highlighted
                - original_size: (width, height) of input image
                - cropped_size: (width, height) of output crop
                - margin: Margin value used
                - min_projection_fraction: Minimum projection fraction used
                - target_aspect_ratio: Target aspect ratio used (if any)
                - output_resolution: Output resolution used (if any)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")

    h, w = image.shape[:2]

    # Create debug context
    debug = DebugContext(debug_dir)
    debug.save("input_original", image)

    # Find the projection using iterative detection
    corners = iterative_detection(image, debug=debug, min_projection_fraction=min_projection_fraction)

    if corners is None:
        raise ValueError("Could not detect projection area in image")

    # Order points consistently
    corners = order_points(corners)
    debug.save_with_corners("detected_corners", image, corners, (255, 255, 0))

    # Clamp corners to image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    # Store original detected corners (for preview)
    corners_detected = corners.copy()

    # Apply margin for the crop (clamped to image bounds)
    corners_for_crop = apply_margin_to_corners(corners, margin, image_size=(w, h))

    # Debug: show both detected and margin-adjusted corners
    if debug.enabled:
        vis = image.copy()
        pts_detected = corners_detected.astype(np.int32).reshape((-1, 1, 2))
        pts_crop = corners_for_crop.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts_detected], True, (0, 0, 255), 2)  # Red = detected
        cv2.polylines(vis, [pts_crop], True, (0, 255, 0), 2)       # Green = with margin
        debug.save("corners_detected_vs_margin", vis)

    # Apply aspect ratio adjustment if target is specified
    target_aspect = None
    if target_aspect_ratio is not None:
        target_aspect = parse_aspect_ratio(target_aspect_ratio)
        corners_for_crop = adjust_corners_for_aspect_ratio(corners_for_crop, target_aspect)
        # Clamp adjusted corners to image bounds
        corners_for_crop[:, 0] = np.clip(corners_for_crop[:, 0], 0, w - 1)
        corners_for_crop[:, 1] = np.clip(corners_for_crop[:, 1], 0, h - 1)

        if debug.enabled:
            vis = image.copy()
            pts_margin = corners_detected.astype(np.int32).reshape((-1, 1, 2))
            pts_aspect = corners_for_crop.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts_margin], True, (0, 0, 255), 2)  # Red = detected
            cv2.polylines(vis, [pts_aspect], True, (255, 0, 255), 2)  # Magenta = aspect-adjusted
            debug.save("corners_aspect_adjusted", vis)

    # Compute output size
    output_size = None
    if target_aspect is not None and output_resolution is not None:
        output_size = compute_target_dimensions(target_aspect, output_resolution)

    # Extract the cropped region
    cropped = warp_perspective_crop(image, corners_for_crop, output_size=output_size)
    debug.save("final_cropped_output", cropped)

    # Normalize coordinates to [0, 1] (using detected corners, not margin-adjusted)
    normalized_corners = []
    for corner in corners_detected:
        x_norm = float(corner[0]) / w
        y_norm = float(corner[1]) / h
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        normalized_corners.append((x_norm, y_norm))

    # Create preview image showing detected region
    preview = image.copy()

    # Draw the detected polygon edges
    pts = corners_detected.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(preview, [pts], True, (0, 0, 255), 3)  # Red polygon

    # Draw the actual crop polygon (may include margin and aspect adjustment)
    pts_crop_final = corners_for_crop.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(preview, [pts_crop_final], True, (0, 255, 0), 2)  # Green polygon (final crop area)

    # Draw corner points
    for i, corner in enumerate(corners_detected):
        pt = (int(corner[0]), int(corner[1]))
        cv2.circle(preview, pt, 10, (0, 0, 255), -1)  # Red filled circle
        cv2.circle(preview, pt, 12, (255, 255, 255), 2)  # White outline

        # Label the corner
        labels = ['TL', 'TR', 'BR', 'BL']
        cv2.putText(
            preview, labels[i],
            (pt[0] + 15, pt[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    info = {
        'corners': normalized_corners,
        'corners_pixel': corners_detected.tolist(),
        'corners_crop_pixel': corners_for_crop.tolist(),
        'preview': preview,
        'original_size': (w, h),
        'cropped_size': (cropped.shape[1], cropped.shape[0]),
        'margin': margin,
        'min_projection_fraction': min_projection_fraction,
        'target_aspect_ratio': target_aspect_ratio,
        'output_resolution': output_resolution,
    }

    return cropped, info


def detect_projection_area(
    image_path: Union[str, Path],
    margin: float = 0.01,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    preview_path: Optional[Union[str, Path]] = None,
    min_projection_fraction: float = 0.10,
    target_aspect_ratio: Optional[str] = None,
    output_resolution: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Detect the projection area in an image file and return cropped image.

    Args:
        image_path: Path to the input image
        margin: Fraction to shrink crop inward (0.02 = 2% smaller than detected region)
        debug: Whether to save debug images (legacy, sets debug_dir to 'debug_output')
        debug_dir: Directory to save debug images with numbered prefixes showing
                   each algorithm step. If None and debug=True, uses 'debug_output'.
        output_path: Optional path to save cropped output image
        preview_path: Optional path to save preview image with detection overlay
        min_projection_fraction: Minimum fraction of image area the projection must cover
                                 (default: 0.10 = 10% of image)
        target_aspect_ratio: Optional target aspect ratio for output (e.g., "16:9", "4:3").
                             If provided, the source region will be center-cropped to achieve
                             exactly this aspect ratio in the output.
        output_resolution: Optional max dimension for output. If provided along with
                           target_aspect_ratio, output will be resized to exact dimensions.

    Returns:
        Tuple of (cropped_image, info_dict):
            - cropped_image: Perspective-corrected crop of the projection area (BGR uint8)
            - info_dict containing:
                - corners: List of 4 (x, y) tuples normalized to [0, 1] (TL, TR, BR, BL)
                - corners_pixel: Corner coordinates in pixels
                - preview: Preview image with detected polygon highlighted
                - original_size: (width, height) of input image
                - cropped_size: (width, height) of output crop
                - margin: Margin value used
                - min_projection_fraction: Minimum projection fraction used
                - target_aspect_ratio: Target aspect ratio used (if any)
                - output_resolution: Output resolution used (if any)
                - input_path: Path to input file
                - output_path: Path to output file (if saved)
                - preview_path: Path to preview file (if saved)
                - debug_dir: Path to debug output directory (if debug enabled)
    """
    image_path = Path(image_path)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Handle debug flag (legacy) vs debug_dir
    if debug and debug_dir is None:
        debug_dir = "debug_output"

    if debug_dir:
        print(f"Debug output will be saved to: {debug_dir}/")

    cropped, info = detect_projection_from_array(
        image,
        margin=margin,
        debug_dir=debug_dir,
        min_projection_fraction=min_projection_fraction,
        target_aspect_ratio=target_aspect_ratio,
        output_resolution=output_resolution,
    )

    if debug_dir:
        info['debug_dir'] = debug_dir

    # Add file path info
    info['input_path'] = str(image_path)

    # Save cropped output if path provided
    if output_path is not None:
        cv2.imwrite(str(output_path), cropped)
        info['output_path'] = str(output_path)

    # Save preview if path provided
    if preview_path is not None:
        cv2.imwrite(str(preview_path), info['preview'])
        info['preview_path'] = str(preview_path)

    return cropped, info


def _main():
    """CLI entry point for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect projection surface in an image'
    )
    parser.add_argument(
        'image_path',
        help='Path to the input image'
    )
    parser.add_argument(
        '-o', '--output',
        default='projection_crop.png',
        help='Output path for cropped image (default: projection_crop.png)'
    )
    parser.add_argument(
        '-p', '--preview',
        default='projection_preview.png',
        help='Output path for preview image (default: projection_preview.png)'
    )
    parser.add_argument(
        '-m', '--margin',
        type=float,
        default=0.01,
        help='Margin to shrink the final crop (0.02 = 2%% smaller)'
    )
    parser.add_argument(
        '--debug',
        nargs='?',
        const='debug_output',
        default=None,
        metavar='DIR',
        help='Save debug images to directory (default: debug_output). '
             'Images are numbered by algorithm step (01_, 02_, etc.)'
    )
    parser.add_argument(
        '--min-projection-fraction',
        type=float,
        default=0.10,
        help='Minimum fraction of image area the projection must cover to be detectable'
    )
    parser.add_argument(
        '-a', '--aspect-ratio',
        type=str,
        default=None,
        metavar='RATIO',
        help='Target aspect ratio for output (e.g., "16:9", "4:3", "1:1"). '
             'The detected region will be center-cropped to achieve this exact ratio.'
    )
    parser.add_argument(
        '-r', '--resolution',
        type=int,
        default=None,
        metavar='PIXELS',
        help='Output resolution (max dimension in pixels, e.g., 1920). '
             'Requires --aspect-ratio. Output will be resized to exact dimensions.'
    )

    args = parser.parse_args()

    # Validate resolution requires aspect-ratio
    if args.resolution is not None and args.aspect_ratio is None:
        parser.error("--resolution requires --aspect-ratio to be specified")

    try:
        cropped, info = detect_projection_area(
            args.image_path,
            margin=args.margin,
            debug_dir=args.debug,
            output_path=args.output,
            preview_path=args.preview,
            min_projection_fraction=args.min_projection_fraction,
            target_aspect_ratio=args.aspect_ratio,
            output_resolution=args.resolution,
        )

        corners = info['corners']

        # Print results
        print("Detected projection corners (normalized coordinates):")
        print(f"  Top-Left:     ({corners[0][0]:.4f}, {corners[0][1]:.4f})")
        print(f"  Top-Right:    ({corners[1][0]:.4f}, {corners[1][1]:.4f})")
        print(f"  Bottom-Right: ({corners[2][0]:.4f}, {corners[2][1]:.4f})")
        print(f"  Bottom-Left:  ({corners[3][0]:.4f}, {corners[3][1]:.4f})")
        print(f"\nOriginal size: {info['original_size']}")
        print(f"Cropped size:  {info['cropped_size']}")
        print(f"Margin:        {info['margin']*100:.1f}%")
        print(f"Min projection fraction: {info['min_projection_fraction']*100:.1f}%")
        if info['target_aspect_ratio']:
            print(f"Target aspect ratio: {info['target_aspect_ratio']}")
        if info['output_resolution']:
            print(f"Output resolution: {info['output_resolution']}")
        print(f"\nCropped image saved to: {args.output}")
        print(f"Preview saved to: {args.preview}")

        if args.debug:
            print(f"Debug images saved to: {args.debug}/")

        # Also print as array for easy copy-paste
        print(f"\nCorners as array: {corners}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(_main())
