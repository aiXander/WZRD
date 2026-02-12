"""
Image alignment for projection mapping.

Aligns a source image (e.g., well-lit surface photo) to match a target image's
geometry using feature matching and homography estimation. Robust to different
lighting conditions through edge-based preprocessing.

Example usage:
    from wzrd import align_images, align_images_file

    # Align arrays directly
    warped, info = align_images(lit_surface, target_crop)

    # Align files with automatic I/O
    warped, info = align_images_file(
        "day_photo.jpg",
        "projection_crop.png",
        output_path="aligned.jpg"
    )
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass, field

DEFAULT_TEMPLATE_SCALES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


@dataclass
class AlignmentResult:
    """Result of image alignment."""
    homography: Optional[np.ndarray]
    num_matches: int
    num_inliers: int
    confidence: float
    method: str  # 'feature' or 'template'


# =============================================================================
# Preprocessing
# =============================================================================

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(gray)


def _compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel operators."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def _preprocess_for_matching(
    img: np.ndarray,
    use_clahe: bool = True,
    clahe_clip_limit: float = 3.0,
    clahe_grid_size: int = 8,
    use_edge_enhancement: bool = True,
    edge_weight: float = 0.5,
    gaussian_blur_size: int = 3,
) -> np.ndarray:
    """Preprocess image for robust feature matching."""
    gray = _to_grayscale(img)

    if use_clahe:
        gray = _apply_clahe(gray, clip_limit=clahe_clip_limit, grid_size=clahe_grid_size)

    if use_edge_enhancement:
        gradient = _compute_gradient_magnitude(gray)
        gray = cv2.addWeighted(gray, 1.0 - edge_weight, gradient, edge_weight, 0)

    if gaussian_blur_size > 1:
        gray = cv2.GaussianBlur(gray, (gaussian_blur_size, gaussian_blur_size), 0)

    return gray


def _extract_edges(gray: np.ndarray, low_threshold: int = 30, high_threshold: int = 100) -> np.ndarray:
    """Extract edges using Canny edge detection."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


# =============================================================================
# Template Matching (Coarse Alignment)
# =============================================================================

def _compute_structure_features(gray: np.ndarray) -> np.ndarray:
    """Compute structural features robust to lighting via gradient magnitude."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def _find_template_location(
    source: np.ndarray,
    template: np.ndarray,
    scales: List[float],
) -> Tuple[Optional[Tuple[int, int, int, int]], float, float]:
    """
    Find template location using multiple methods and voting.

    Returns:
        (x, y, w, h) bounding box, best scale, and match score
    """
    source_gray = _to_grayscale(source)
    template_gray = _to_grayscale(template)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    source_clahe = clahe.apply(source_gray)
    template_clahe = clahe.apply(template_gray)

    source_edge = _extract_edges(source_clahe, 30, 100)
    template_edge = _extract_edges(template_clahe, 30, 100)

    kernel = np.ones((3, 3), np.uint8)
    source_edge = cv2.dilate(source_edge, kernel, iterations=1)
    template_edge = cv2.dilate(template_edge, kernel, iterations=1)
    source_edge = cv2.GaussianBlur(source_edge, (5, 5), 2)
    template_edge = cv2.GaussianBlur(template_edge, (5, 5), 2)

    source_struct = _compute_structure_features(source_clahe)
    template_struct = _compute_structure_features(template_clahe)

    template_h, template_w = template_gray.shape[:2]
    source_h, source_w = source_gray.shape[:2]

    candidates = []

    methods = [
        (cv2.TM_CCOEFF_NORMED, source_edge, template_edge, 1.0),
        (cv2.TM_CCORR_NORMED, source_edge, template_edge, 0.8),
        (cv2.TM_CCOEFF_NORMED, source_struct, template_struct, 0.7),
        (cv2.TM_CCOEFF_NORMED, source_clahe, template_clahe, 0.5),
    ]

    for scale in scales:
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)

        if new_w >= source_w or new_h >= source_h:
            continue
        if new_w < 50 or new_h < 50:
            continue

        for method, src_img, tpl_img, weight in methods:
            resized_tpl = cv2.resize(tpl_img, (new_w, new_h))
            result = cv2.matchTemplate(src_img, resized_tpl, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            candidates.append({
                'bbox': (max_loc[0], max_loc[1], new_w, new_h),
                'scale': scale,
                'score': max_val * weight,
                'raw_score': max_val,
            })

    if not candidates:
        return None, 1.0, 0.0

    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Cluster nearby candidates and vote
    vote_radius = 50
    votes = {}
    for c in candidates[:20]:
        x, y, w, h = c['bbox']
        grid_x = x // vote_radius
        grid_y = y // vote_radius
        key = (grid_x, grid_y, c['scale'])
        if key not in votes:
            votes[key] = {'count': 0, 'total_score': 0, 'best': c}
        votes[key]['count'] += 1
        votes[key]['total_score'] += c['score']
        if c['score'] > votes[key]['best']['score']:
            votes[key]['best'] = c

    best_vote = max(votes.values(), key=lambda v: v['count'] * v['total_score'])
    best = best_vote['best']

    return best['bbox'], best['scale'], best['raw_score']


# =============================================================================
# Feature-Based Alignment
# =============================================================================

def _find_homography_feature_based(
    source: np.ndarray,
    target: np.ndarray,
    max_features: int = 20000,
    ratio_threshold: float = 0.65,
) -> Optional[AlignmentResult]:
    """Find homography using direct feature matching on full images."""
    best_result = None
    best_inliers = 0

    detectors_to_try = [
        ("SIFT", cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2, 0.65),
        ("AKAZE", cv2.AKAZE_create(), cv2.NORM_HAMMING, 0.75),
    ]

    source_gray = _to_grayscale(source)
    target_gray = _to_grayscale(target)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    source_clahe = clahe.apply(source_gray)
    target_clahe = clahe.apply(target_gray)

    preprocessing_options = [
        (source_clahe, target_clahe),
        (source_gray, target_gray),
    ]

    for det_name, detector, norm_type, ratio_thresh in detectors_to_try:
        for proc_source, proc_target in preprocessing_options:
            kp1, desc1 = detector.detectAndCompute(proc_source, None)
            kp2, desc2 = detector.detectAndCompute(proc_target, None)

            if desc1 is None or desc2 is None or len(kp1) < 50 or len(kp2) < 50:
                continue

            bf = cv2.BFMatcher(norm_type)
            matches = bf.knnMatch(desc1, desc2, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 8:
                continue

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=5000,
                confidence=0.999
            )

            if H is None:
                continue

            inliers = int(np.sum(mask))
            confidence = inliers / len(good_matches)

            if inliers > best_inliers:
                best_inliers = inliers
                best_result = AlignmentResult(
                    homography=H,
                    num_matches=len(good_matches),
                    num_inliers=inliers,
                    confidence=confidence,
                    method='feature',
                )

    return best_result


def _validate_homography(H: np.ndarray, source_shape: Tuple, target_shape: Tuple) -> bool:
    """Check if homography produces reasonable transformation."""
    src_h, src_w = source_shape[:2]
    dst_h, dst_w = target_shape[:2]

    src_corners = np.float32([
        [0, 0], [src_w, 0], [src_w, src_h], [0, src_h]
    ]).reshape(-1, 1, 2)

    dst_corners = cv2.perspectiveTransform(src_corners, H)
    dst_corners = dst_corners.reshape(-1, 2)

    margin = max(dst_w, dst_h) * 2
    if np.any(dst_corners < -margin) or np.any(dst_corners > margin + max(dst_w, dst_h)):
        return False

    def cross2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    v1 = dst_corners[1] - dst_corners[0]
    v2 = dst_corners[2] - dst_corners[1]
    v3 = dst_corners[3] - dst_corners[2]
    v4 = dst_corners[0] - dst_corners[3]

    crosses = np.array([cross2d(v1, v2), cross2d(v2, v3), cross2d(v3, v4), cross2d(v4, v1)])
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        return False

    return True


def _compute_homography_from_corners(
    src_corners: np.ndarray,
    dst_size: Tuple[int, int]
) -> np.ndarray:
    """Compute homography from source corners to full destination image."""
    dst_w, dst_h = dst_size
    dst_corners = np.float32([
        [0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]
    ])
    return cv2.getPerspectiveTransform(src_corners.astype(np.float32), dst_corners)


def _refine_with_ecc(
    source: np.ndarray,
    template: np.ndarray,
    initial_warp: np.ndarray,
    num_iterations: int = 100,
    termination_eps: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """Refine alignment using Enhanced Correlation Coefficient (ECC)."""
    source_gray = _to_grayscale(source)
    template_gray = _to_grayscale(template)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    source_gray = clahe.apply(source_gray)
    template_gray = clahe.apply(template_gray)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iterations, termination_eps)
    warp_matrix = initial_warp.copy().astype(np.float32)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            template_gray, source_gray, warp_matrix,
            cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
        return warp_matrix, cc
    except cv2.error:
        return initial_warp, 0.0


def _generate_alignment_overlay(warped: np.ndarray) -> np.ndarray:
    """Generate a high-contrast edge image for projector alignment verification.

    Produces prominent edges blended with a faint version of the original image.
    Only major structural edges are kept (walls, doors, large objects) while
    small texture noise is suppressed.
    """
    gray = _to_grayscale(warped)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Moderate blur to suppress small texture while keeping structural edges
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 2.0)

    # Canny with raised thresholds to favor strong edges
    edges = cv2.Canny(blurred, 50, 150)

    # Remove small isolated edge fragments
    kernel_clean = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_clean, iterations=1)

    # Dilate edges for visibility when projected
    kernel_dilate = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    # Blend: bright edges over a faint version of the original for context
    base = (gray * 0.2).astype(np.uint8)
    overlay = cv2.max(base, edges)

    return overlay


def _warp_image(
    src: np.ndarray,
    homography: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """Warp source image using homography to match target size."""
    return cv2.warpPerspective(
        src, homography, target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


# =============================================================================
# Main Alignment Function
# =============================================================================

def align_images(
    source: np.ndarray,
    target: np.ndarray,
    max_features: int = 10000,
    ratio_threshold: float = 0.75,
    template_scales: Optional[List[float]] = None,
    use_ecc_refinement: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, AlignmentResult]:
    """
    Align source image to match target geometry.

    Uses a multi-stage approach:
    1. Direct feature matching (most robust for good matches)
    2. Template matching fallback (for difficult cases)
    3. ECC refinement for sub-pixel accuracy

    Args:
        source: Source image (BGR, uint8)
        target: Target image to align to (BGR, uint8)
        max_features: Maximum features for detection
        ratio_threshold: Lowe's ratio test threshold
        template_scales: Scales to try for template matching
        use_ecc_refinement: Whether to apply ECC refinement
        verbose: Print progress messages

    Returns:
        Tuple of (warped_image, AlignmentResult)
    """
    if template_scales is None:
        template_scales = DEFAULT_TEMPLATE_SCALES

    target_h, target_w = target.shape[:2]
    source_h, source_w = source.shape[:2]

    if verbose:
        print(f"Source: {source.shape}, Target: {target.shape}")

    # Stage 1: Direct feature-based matching
    if verbose:
        print("[Stage 1] Feature-based matching...")

    feature_result = _find_homography_feature_based(
        source, target, max_features=max_features, ratio_threshold=ratio_threshold
    )

    use_feature_result = False
    if feature_result is not None and feature_result.homography is not None:
        if feature_result.confidence > 0.35 and feature_result.num_inliers >= 8:
            if _validate_homography(feature_result.homography, source.shape, target.shape):
                if verbose:
                    print(f"  Feature matching: {feature_result.num_inliers} inliers, "
                          f"{feature_result.confidence:.1%} confidence")
                use_feature_result = True
                refined_H = feature_result.homography
                result = feature_result

    if not use_feature_result:
        # Stage 2: Template matching fallback
        if verbose:
            print("[Stage 2] Template matching fallback...")

        scales = list(np.arange(0.2, 1.5, 0.05))
        bbox, best_scale, template_score = _find_template_location(source, target, scales)

        if bbox is None:
            if verbose:
                print("  Template matching failed, using identity")
            bbox = (0, 0, source_w, source_h)
            best_scale = 1.0
            template_score = 0.0
        else:
            if verbose:
                print(f"  Template: scale={best_scale:.2f}, score={template_score:.3f}")

        x, y, w, h = bbox
        src_corners = np.float32([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ])
        refined_H = _compute_homography_from_corners(src_corners, (target_w, target_h))

        result = AlignmentResult(
            homography=refined_H,
            num_matches=0,
            num_inliers=0,
            confidence=template_score,
            method='template',
        )

    # Stage 3: ECC refinement
    if use_ecc_refinement:
        if verbose:
            print("[Stage 3] ECC refinement...")
        try:
            warped_initial = _warp_image(source, refined_H, (target_w, target_h))
            final_H, ecc_score = _refine_with_ecc(warped_initial, target, np.eye(3))

            if ecc_score > 0.3:
                if verbose:
                    print(f"  ECC correlation: {ecc_score:.3f}")
                refined_H = final_H @ refined_H
        except Exception as e:
            if verbose:
                print(f"  ECC failed: {e}")

    # Final warp
    warped = _warp_image(source, refined_H, (target_w, target_h))
    result.homography = refined_H

    return warped, result


def align_images_file(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    max_features: int = 10000,
    ratio_threshold: float = 0.75,
    template_scales: Optional[List[float]] = None,
    use_ecc_refinement: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Align source image file to match target image geometry.

    Args:
        source_path: Path to source image (e.g., well-lit surface photo)
        target_path: Path to target image to align to
        output_path: Optional path to save aligned output
        max_features: Maximum features for detection
        ratio_threshold: Lowe's ratio test threshold
        template_scales: Scales to try for template matching
        use_ecc_refinement: Whether to apply ECC refinement
        verbose: Print progress messages

    Returns:
        Tuple of (warped_image, info_dict)
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    source = cv2.imread(str(source_path))
    if source is None:
        raise FileNotFoundError(f"Could not load source image: {source_path}")

    target = cv2.imread(str(target_path))
    if target is None:
        raise FileNotFoundError(f"Could not load target image: {target_path}")

    warped, result = align_images(
        source, target,
        max_features=max_features,
        ratio_threshold=ratio_threshold,
        template_scales=template_scales,
        use_ecc_refinement=use_ecc_refinement,
        verbose=verbose,
    )

    info = {
        'source_path': str(source_path),
        'target_path': str(target_path),
        'source_size': (source.shape[1], source.shape[0]),
        'target_size': (target.shape[1], target.shape[0]),
        'method': result.method,
        'num_matches': result.num_matches,
        'num_inliers': result.num_inliers,
        'confidence': result.confidence,
        'homography': result.homography,
    }

    if output_path is not None:
        output_path = Path(output_path)
        cv2.imwrite(str(output_path), warped)
        info['output_path'] = str(output_path)
        if verbose:
            print(f"Saved: {output_path}")

        # Generate and save alignment overlay for projector calibration
        overlay = _generate_alignment_overlay(warped)
        overlay_path = output_path.with_name(
            output_path.stem + "_edges" + ".png"
        )
        cv2.imwrite(str(overlay_path), overlay)
        info['overlay_path'] = str(overlay_path)
        if verbose:
            print(f"Saved alignment overlay: {overlay_path}")

    return warped, info


# =============================================================================
# CLI
# =============================================================================

def _main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Align a source image to match target geometry."
    )
    parser.add_argument(
        "source", help="Path to source image (e.g., well-lit surface photo)"
    )
    parser.add_argument(
        "target", help="Path to target image to align to"
    )
    parser.add_argument(
        "-o", "--output", default="aligned.jpg",
        help="Output path (default: aligned.jpg)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print progress messages"
    )

    args = parser.parse_args()

    try:
        warped, info = align_images_file(
            args.source, args.target,
            output_path=args.output,
            verbose=args.verbose,
        )

        print(f"Alignment complete!")
        print(f"  Method: {info['method']}")
        print(f"  Confidence: {info['confidence']:.2%}")
        if info['num_matches'] > 0:
            print(f"  Inliers: {info['num_inliers']}/{info['num_matches']}")
        print(f"  Output: {args.output}")
        if 'overlay_path' in info:
            print(f"  Alignment overlay: {info['overlay_path']}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(_main())
