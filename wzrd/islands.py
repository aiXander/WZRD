"""
Island extraction from color-segmented images.

Functions to find and crop connected components (islands) from
color-segmented images, where each distinct color represents a separate region.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Literal
Connectivity = Literal[4, 8]


def _to_cielab(image: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to float CIELAB (L* 0-100, a*/b* -128..127)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def _bgr_to_cielab_pixel(bgr: Tuple[int, int, int]) -> np.ndarray:
    """Convert a single BGR color to float CIELAB."""
    u8 = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0]
    return np.array([
        u8[0] * 100.0 / 255.0,
        u8[1] - 128.0,
        u8[2] - 128.0,
    ], dtype=np.float32)


def _lab_centers_to_bgr(lab_centers: np.ndarray) -> np.ndarray:
    """Convert array of float CIELAB values (N,3) to BGR uint8 (N,3)."""
    u8 = np.empty((len(lab_centers), 1, 3), dtype=np.uint8)
    u8[:, 0, 0] = np.clip(lab_centers[:, 0] * 255.0 / 100.0, 0, 255).astype(np.uint8)
    u8[:, 0, 1] = np.clip(lab_centers[:, 1] + 128.0, 0, 255).astype(np.uint8)
    u8[:, 0, 2] = np.clip(lab_centers[:, 2] + 128.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)


def detect_colors(
    image: np.ndarray,
    min_pixel_fraction: float = 0.001,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    background_threshold: float = 15.0,
    delta_e_threshold: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect distinct colors using greedy clustering in CIELAB color space.

    Automatically discovers the number of colors — no fixed cluster count
    needed.  Colors within ``delta_e_threshold`` (approximate CIE76 ΔE) of
    each other are merged into one cluster.  Works well for AI-generated
    flat-color images with small intra-region variation.

    Args:
        image: BGR image
        min_pixel_fraction: Minimum fraction of total pixels for a color
        background_color: Background color to exclude (BGR)
        background_threshold: ΔE from background below which a pixel is
            treated as background and excluded from clustering
        delta_e_threshold: ΔE threshold for merging similar colors.
            Lower = more sensitive to subtle color differences.

    Returns:
        Tuple of:
        - Array of detected colors (BGR, excluding background)
        - Foreground mask (bool ndarray, True = foreground pixel)
    """
    total_pixels = image.shape[0] * image.shape[1]
    min_pixel_count = int(total_pixels * min_pixel_fraction)

    # Convert to CIELAB (perceptually uniform)
    lab = _to_cielab(image)
    bg_lab = _bgr_to_cielab_pixel(background_color)

    # Foreground mask — pixels far enough from background
    delta_from_bg = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=2))
    fg_mask = delta_from_bg > background_threshold

    fg_lab = lab[fg_mask]
    if len(fg_lab) == 0:
        return np.array([], dtype=np.uint8).reshape(0, 3), fg_mask

    # Light quantisation in LAB to reduce unique-pixel count for speed.
    # Clamp to delta_e_threshold so quantisation never exceeds clustering radius.
    q_step = min(2.0, delta_e_threshold) if delta_e_threshold > 0 else 2.0
    fg_lab_q = np.round(fg_lab / q_step).astype(np.int32)
    unique_q, counts = np.unique(fg_lab_q, axis=0, return_counts=True)
    unique_lab = unique_q.astype(np.float32) * q_step

    # Greedy clustering: largest group first, absorb within threshold
    assigned = np.zeros(len(unique_lab), dtype=bool)
    clusters: List[Tuple[np.ndarray, int]] = []

    for idx in np.argsort(-counts):
        if assigned[idx]:
            continue
        seed = unique_lab[idx]
        dists = np.sqrt(np.sum((unique_lab - seed) ** 2, axis=1))
        in_cluster = (~assigned) & (dists <= delta_e_threshold)
        if not np.any(in_cluster):
            continue

        c_counts = counts[in_cluster]
        c_colors = unique_lab[in_cluster]
        total = int(np.sum(c_counts))
        center = np.sum(c_colors * c_counts[:, np.newaxis], axis=0) / total

        assigned[in_cluster] = True
        clusters.append((center, total))

    # Filter small clusters, sort largest first
    clusters = [(c, n) for c, n in clusters if n >= min_pixel_count]
    clusters.sort(key=lambda x: -x[1])

    if not clusters:
        return np.array([], dtype=np.uint8).reshape(0, 3), fg_mask

    lab_centers = np.array([c for c, _ in clusters], dtype=np.float32)
    return _lab_centers_to_bgr(lab_centers), fg_mask


def quantize_to_colors(
    image: np.ndarray,
    colors: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Quantize image pixels to the nearest color from a palette.

    Args:
        image: BGR image
        colors: Array of colors to quantize to (BGR)
        background_color: Background color (BGR)

    Returns:
        Quantized image
    """
    # Add background to palette
    palette = np.vstack([np.array([background_color], dtype=np.uint8), colors])

    # Reshape for distance calculation
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Find nearest color for each pixel
    distances = np.linalg.norm(
        pixels[:, np.newaxis, :] - palette[np.newaxis, :, :].astype(np.float32),
        axis=2
    )
    nearest_idx = np.argmin(distances, axis=1)

    # Map to colors
    quantized = palette[nearest_idx].reshape(image.shape)

    return quantized


def extract_color_mask(
    image: np.ndarray,
    color: np.ndarray,
    tolerance: int = 10,
) -> np.ndarray:
    """
    Extract a binary mask for pixels matching a specific color.

    Args:
        image: BGR image
        color: Target color (BGR)
        tolerance: Color matching tolerance

    Returns:
        Binary mask (255 for matching pixels, 0 otherwise)
    """
    # Calculate distance from target color
    diff = np.abs(image.astype(np.int32) - color.astype(np.int32))
    distance = np.sum(diff, axis=2)

    # Create mask
    mask = (distance <= tolerance * 3).astype(np.uint8) * 255

    return mask


def find_islands(
    image: Union[np.ndarray, str, Path],
    min_area: int = 0,
    connectivity: Connectivity = 8,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Find connected components (islands) in a binary image.

    Args:
        image: Binary image (white regions on black) as numpy array or path
        min_area: Minimum area in pixels to include (filters noise)
        connectivity: 4 (orthogonal) or 8 (diagonal) pixel connectivity

    Returns:
        Tuple of:
        - List of island info dicts with keys: id, x, y, width, height, area_pixels, centroid
        - Labels image (same size as input, each pixel labeled with island ID)
        - Binary image that was processed
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image}")
    else:
        img = image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure binary (clean up compression artifacts)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=connectivity
    )

    islands = []
    # Skip label 0 (background)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        island_info = {
            'id': i,
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'area_pixels': int(area),
            'centroid': (float(centroids[i, 0]), float(centroids[i, 1])),
        }
        islands.append(island_info)

    return islands, labels_im, binary_img


def crop_island(
    image: np.ndarray,
    island: Dict,
) -> np.ndarray:
    """
    Crop a single island from the image.

    Args:
        image: Source image (grayscale or color)
        island: Island info dict with x, y, width, height

    Returns:
        Cropped image region
    """
    x, y = island['x'], island['y']
    w, h = island['width'], island['height']
    return image[y:y+h, x:x+w].copy()


def crop_island_as_mask(
    labels_im: np.ndarray,
    island: Dict,
) -> np.ndarray:
    """
    Crop a single island from labels image as a white mask.

    Args:
        labels_im: Labels image from connected components
        island: Island info dict with id, x, y, width, height

    Returns:
        Binary mask of the island (white on black)
    """
    x, y = island['x'], island['y']
    w, h = island['width'], island['height']

    # Crop the region
    region = labels_im[y:y+h, x:x+w]

    # Create mask for this specific island
    mask = (region == island['id']).astype(np.uint8) * 255

    return mask


def extract_color_regions(
    image: Union[np.ndarray, str, Path],
    output_dir: Union[str, Path],
    max_colors: int = 8,
    min_area_fraction: float = 0.01,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    quantize: bool = True,
    output_min_size: Optional[int] = 512,
    surface: Optional[Union[np.ndarray, str, Path]] = None,
    delta_e_threshold: float = 5.0,
    background_threshold: float = 15.0,
    merge_same_color: bool = True,
) -> Tuple[List[Dict], Path]:
    """
    Extract color regions from a color-segmented image.

    Colors are discovered automatically via greedy clustering in CIELAB
    space — no need to know the number of regions upfront.  Each spatially
    connected component of each detected color becomes its own output region.

    Primary outputs per region:
        ``region_mask_NNN_color_RRGGBB.png`` — full-resolution binary mask
        (white pixels = this region, at source image resolution).

    Secondary aid outputs per region:
        ``region_crop_NNN_color_RRGGBB.png`` — tight crop of the mask,
        rescaled to ``output_min_size``.

    Args:
        image: Color-segmented image (BGR) as numpy array or path
        output_dir: Directory to save region masks and crops
        max_colors: Upper cap on number of distinct colors to keep
        min_area_fraction: Minimum area as fraction of total image (0.0-0.5)
        background_color: Background color to exclude (BGR)
        quantize: Whether to quantize image to clean up artifacts
        output_min_size: Target size for min(width, height) of cropped aid
                         outputs.  Set to None to disable crop generation.
        surface: Optional surface image to extract corresponding crops from.
                 If resolution differs from input, the input is resized to match
                 the surface resolution (surface is treated as ground truth).
        delta_e_threshold: CIELAB ΔE threshold for merging similar colors.
                           Lower = more sensitive to subtle differences.
        background_threshold: CIELAB ΔE from background below which pixels
                              are treated as background.
        merge_same_color: If True, all disconnected blobs of the same color
                          are combined into a single region. If False, each
                          spatially connected component is a separate region.

    Returns:
        Tuple of (list of region info dicts, output directory path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image if path provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image}")
    else:
        img = image.copy()

    original_shape = img.shape[:2]

    # Load and prepare surface image if provided
    surface_img = None
    if surface is not None:
        if isinstance(surface, (str, Path)):
            surface_img = cv2.imread(str(surface), cv2.IMREAD_COLOR)
            if surface_img is None:
                raise FileNotFoundError(f"Could not load surface image: {surface}")
        else:
            surface_img = surface.copy()

        # Check if surface resolution matches input; resize input to match surface
        surface_h, surface_w = surface_img.shape[:2]
        input_h, input_w = original_shape
        if (surface_w, surface_h) != (input_w, input_h):
            print(f"Resizing input image from {input_w}x{input_h} to {surface_w}x{surface_h} (matching surface)")
            img = cv2.resize(img, (surface_w, surface_h), interpolation=cv2.INTER_LINEAR)
            original_shape = img.shape[:2]

    total_pixels = original_shape[0] * original_shape[1]
    min_area_pixels = int(total_pixels * min_area_fraction)

    # Detect colors (auto-discovers count via CIELAB greedy clustering)
    colors, fg_mask = detect_colors(
        img,
        min_pixel_fraction=min_area_fraction,
        background_color=background_color,
        background_threshold=background_threshold,
        delta_e_threshold=delta_e_threshold,
    )
    colors = colors[:max_colors]

    if quantize:
        img = quantize_to_colors(img, colors, background_color)

    # Force near-background pixels to exact background color
    bg = np.array(background_color, dtype=np.uint8)
    img[~fg_mask] = bg

    # Save quantized image for reference
    cv2.imwrite(str(output_dir / 'quantized.png'), img)

    # After quantization, pixels should be exact matches (tolerance=0)
    # Without quantization, allow some tolerance for compression artifacts
    color_tolerance = 0 if quantize else 10

    # Morphological kernel for cleaning thin artifacts
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    img_h, img_w = original_shape
    all_regions = []

    # Process each color.  When merge_same_color is True the whole color
    # mask is one region; otherwise split by connected components.
    region_counter = 0
    for color_idx, color in enumerate(colors):
        # Extract mask for this color
        mask = extract_color_mask(img, color, tolerance=color_tolerance)

        # Clean up thin edge artifacts with morphological opening
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)

        if merge_same_color:
            # Treat all pixels of this color as a single region
            blobs: List[Tuple[np.ndarray, int, int, int, int, int]] = []
            area_pixels = int(np.count_nonzero(mask))
            if area_pixels < min_area_pixels:
                continue
            ys, xs = np.where(mask > 0)
            x_min, y_min = int(xs.min()), int(ys.min())
            x_max, y_max = int(xs.max()), int(ys.max())
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            blobs.append((mask, area_pixels, x_min, y_min, width, height))
        else:
            # Connected components — each spatial blob is its own island
            num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            blobs = []
            for cc_id in range(1, num_cc):
                area_pixels = int(cc_stats[cc_id, cv2.CC_STAT_AREA])
                if area_pixels < min_area_pixels:
                    continue
                full_mask = (cc_labels == cc_id).astype(np.uint8) * 255
                blobs.append((
                    full_mask, area_pixels,
                    int(cc_stats[cc_id, cv2.CC_STAT_LEFT]),
                    int(cc_stats[cc_id, cv2.CC_STAT_TOP]),
                    int(cc_stats[cc_id, cv2.CC_STAT_WIDTH]),
                    int(cc_stats[cc_id, cv2.CC_STAT_HEIGHT]),
                ))

        for full_mask, area_pixels, x_min, y_min, width, height in blobs:
            color_hex = f"{color[2]:02x}{color[1]:02x}{color[0]:02x}"  # BGR to RGB hex
            region_counter += 1

            # --- Primary output: full-resolution mask ---
            mask_filename = f"region_mask_{region_counter:03d}_color_{color_hex}.png"
            cv2.imwrite(str(output_dir / mask_filename), full_mask)

            # Round source crop dimensions down to multiples of 8
            crop_w = (width // 8) * 8
            crop_h = (height // 8) * 8

            region_info = {
                'region_mask': mask_filename,
                'source_box': {
                    'x': x_min,
                    'y': y_min,
                    'width': int(crop_w) if crop_w >= 8 else int(width),
                    'height': int(crop_h) if crop_h >= 8 else int(height),
                },
                'area_pixels': area_pixels,
            }

            # --- Secondary aid output: tight crop (rescaled) ---
            if output_min_size is not None and crop_w >= 8 and crop_h >= 8:
                crop = full_mask[y_min:y_min+crop_h, x_min:x_min+crop_w].copy()

                out_w, out_h = crop_w, crop_h
                min_dim = min(crop_w, crop_h)
                if min_dim > 0:
                    scale = output_min_size / min_dim
                    out_w = int(round(crop_w * scale))
                    out_h = int(round(crop_h * scale))

                # Ensure dimensions are multiples of 8
                out_w = max(8, (out_w // 8) * 8)
                out_h = max(8, (out_h // 8) * 8)

                if (out_w, out_h) != (crop_w, crop_h):
                    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    _, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)

                crop_filename = f"region_crop_{region_counter:03d}_color_{color_hex}.png"
                cv2.imwrite(str(output_dir / crop_filename), crop)
                region_info['crop_filename'] = crop_filename

                # Surface crop (same bbox / scale as the mask crop)
                if surface_img is not None:
                    surface_crop = surface_img[y_min:y_min+crop_h, x_min:x_min+crop_w].copy()
                    if (out_w, out_h) != (crop_w, crop_h):
                        surface_crop = cv2.resize(surface_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    surface_filename = f"region_crop_{region_counter:03d}_color_{color_hex}_surface.png"
                    cv2.imwrite(str(output_dir / surface_filename), surface_crop)
                    region_info['surface_filename'] = surface_filename

            all_regions.append(region_info)

    # Save metadata - minimal info needed for reconstruction
    metadata = {
        'source_resolution': [original_shape[1], original_shape[0]],  # [width, height]
        'regions': all_regions,
    }

    json_path = output_dir / 'islands.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return all_regions, output_dir


def reassemble_regions(
    metadata_path: Union[str, Path],
    processed_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Reassemble processed regions back into the original image layout.

    Args:
        metadata_path: Path to islands.json
        processed_dir: Directory containing processed region images
                       (default: same as metadata)
        output_path: Path to save the reassembled image (optional)

    Returns:
        Reassembled image
    """
    metadata_path = Path(metadata_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if processed_dir is None:
        processed_dir = metadata_path.parent
    else:
        processed_dir = Path(processed_dir)

    # Create output image - support both old and new format
    if 'source_resolution' in metadata:
        canvas_w, canvas_h = metadata['source_resolution']
    else:
        size = metadata['source_image_size']
        canvas_w, canvas_h = size['width'], size['height']

    output = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Support both old 'islands' and new 'regions' key
    regions = metadata.get('regions', metadata.get('islands', []))

    # Place each region
    for region in regions:
        filename = region['filename']
        region_img = cv2.imread(str(processed_dir / filename), cv2.IMREAD_GRAYSCALE)

        if region_img is None:
            print(f"Warning: Could not load {filename}")
            continue

        # Support both old and new format for box coordinates
        if 'source_box' in region:
            box = region['source_box']
            x, y = box['x'], box['y']
            target_w, target_h = box['width'], box['height']
        else:
            x, y = region['x'], region['y']
            target_w, target_h = region['width'], region['height']

        # Resize animated crop to fit the original source box
        region_img = cv2.resize(
            region_img,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )
        # Re-threshold to keep it binary after interpolation
        _, region_img = cv2.threshold(region_img, 127, 255, cv2.THRESH_BINARY)

        # Clamp to canvas bounds
        place_h = min(target_h, canvas_h - y)
        place_w = min(target_w, canvas_w - x)

        # Blend (max) to handle overlapping regions
        existing = output[y:y+place_h, x:x+place_w]
        output[y:y+place_h, x:x+place_w] = np.maximum(
            existing,
            region_img[:place_h, :place_w]
        )

    if output_path:
        cv2.imwrite(str(output_path), output)

    return output


def crop_islands(
    image: Union[np.ndarray, str, Path],
    min_area: int = 0,
    connectivity: Connectivity = 8,
) -> List[Tuple[Dict, np.ndarray]]:
    """
    Find and crop all islands from a binary image.

    Args:
        image: Binary image (white regions on black)
        min_area: Minimum area in pixels
        connectivity: 4 or 8 pixel connectivity

    Returns:
        List of (island_info, cropped_image) tuples
    """
    islands, labels_im, binary_img = find_islands(image, min_area, connectivity)

    results = []
    for island in islands:
        crop = crop_island(binary_img, island)
        results.append((island, crop))

    return results


def save_islands(
    image: Union[np.ndarray, str, Path],
    output_dir: Union[str, Path],
    min_area: int = 0,
    connectivity: Connectivity = 8,
    save_json: bool = True,
) -> Tuple[List[Dict], Path]:
    """
    Find islands and save each as a separate image file.

    Args:
        image: Binary image (white regions on black)
        output_dir: Directory to save cropped islands
        min_area: Minimum area in pixels
        connectivity: 4 or 8 pixel connectivity
        save_json: Whether to save a JSON file with island metadata

    Returns:
        Tuple of (list of island info dicts with 'filename' added, output directory path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    islands, labels_im, binary_img = find_islands(image, min_area, connectivity)

    for island in islands:
        crop = crop_island(binary_img, island)
        filename = f"island_{island['id']}.png"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), crop)
        island['filename'] = filename

    if save_json:
        json_path = output_dir / 'islands.json'
        with open(json_path, 'w') as f:
            json.dump(islands, f, indent=2)

    return islands, output_dir


def crop_islands_file(
    input_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    min_area: int = 0,
    connectivity: Connectivity = 8,
) -> Tuple[List[Dict], Path]:
    """
    Process an image file and save island crops.

    Args:
        input_path: Path to binary segmentation map image
        output_dir: Output directory (default: same folder as input + '_islands')
        min_area: Minimum area in pixels
        connectivity: 4 or 8 pixel connectivity

    Returns:
        Tuple of (list of island info dicts, output directory path)
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_islands"

    return save_islands(
        str(input_path),
        output_dir,
        min_area=min_area,
        connectivity=connectivity,
    )


def _cli():
    """CLI entry point for ``python -m wzrd.islands``."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract color regions from segmented images'
    )
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('--max-colors', type=int, default=8,
                        help='Upper cap on number of distinct colors (default: 8)')
    parser.add_argument('--min-area', type=float, default=0.01,
                        help='Minimum region area as fraction of image (default: 0.01)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Skip quantization step')
    parser.add_argument('--output-min-size', type=int, default=512,
                        help='Rescale crops so min(w,h)=this value (default: 512, use 0 to disable)')
    parser.add_argument('--delta-e', type=float, default=15.0,
                        help='CIELAB deltaE threshold for merging similar colors (default: 15.0)')
    parser.add_argument('--bg-threshold', type=float, default=15.0,
                        help='CIELAB deltaE from background to treat as background (default: 15.0)')
    parser.add_argument('--surface', help='Optional surface image to extract corresponding crops from')
    parser.add_argument('--no-merge', action='store_true',
                        help='Split same-color disconnected blobs into separate regions')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = args.output or input_path.parent / f"{input_path.stem}_regions"

    # Convert 0 to None to disable rescaling
    output_min_size = args.output_min_size if args.output_min_size > 0 else None

    regions, out_path = extract_color_regions(
        input_path,
        output_dir,
        max_colors=args.max_colors,
        min_area_fraction=args.min_area,
        quantize=not args.no_quantize,
        output_min_size=output_min_size,
        surface=args.surface,
        delta_e_threshold=args.delta_e,
        background_threshold=args.bg_threshold,
        merge_same_color=not args.no_merge,
    )

    print(f"Extracted {len(regions)} color regions to {out_path}")
    for region in regions:
        box = region['source_box']
        surface_info = f" + {region['surface_filename']}" if 'surface_filename' in region else ""
        print(f"  - {region['region_mask']}{surface_info}: source_box ({box['x']}, {box['y']}, {box['width']}x{box['height']})")


if __name__ == '__main__':
    _cli()
