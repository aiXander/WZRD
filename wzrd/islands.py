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
from sklearn.cluster import KMeans


Connectivity = Literal[4, 8]


def detect_colors(
    image: np.ndarray,
    max_colors: int = 5,
    min_pixel_fraction: float = 0.001,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect distinct colors in an image using K-means clustering.

    Args:
        image: BGR image
        max_colors: Maximum number of colors to detect
        min_pixel_fraction: Minimum fraction of total pixels for a color to be valid
        background_color: Background color to exclude (BGR)

    Returns:
        Tuple of:
        - Array of detected colors (BGR, excluding background)
        - Quantized image with pixels snapped to nearest cluster center
    """
    total_pixels = image.shape[0] * image.shape[1]
    min_pixel_count = int(total_pixels * min_pixel_fraction)

    # Reshape image to list of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Use K-means to find color clusters
    kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)

    # Count pixels per cluster
    unique, counts = np.unique(labels, return_counts=True)

    # Filter out background and small clusters
    valid_colors = []
    bg = np.array(background_color, dtype=np.uint8)

    for cluster_id, count in zip(unique, counts):
        color = centers[cluster_id]
        # Check if this is background (within tolerance)
        if np.linalg.norm(color.astype(float) - bg.astype(float)) < 30:
            continue
        if count >= min_pixel_count:
            valid_colors.append(color)

    # Create quantized image
    quantized = centers[labels].reshape(image.shape)

    return np.array(valid_colors), quantized


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
    max_colors: int = 5,
    min_area_fraction: float = 0.05,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    quantize: bool = True,
    output_min_size: Optional[int] = 512,
    surface: Optional[Union[np.ndarray, str, Path]] = None,
) -> Tuple[List[Dict], Path]:
    """
    Extract color regions from a color-segmented image.

    Each distinct color (except background) becomes a single white mask crop,
    containing ALL pixels of that color (even if disconnected).

    Args:
        image: Color-segmented image (BGR) as numpy array or path
        output_dir: Directory to save cropped regions
        max_colors: Maximum number of colors to detect
        min_area_fraction: Minimum area as fraction of total image (0.0-1.0)
        background_color: Background color to exclude (BGR)
        quantize: Whether to quantize image to clean up artifacts
        output_min_size: Target size for min(width, height) of output crops.
                         Crops are rescaled to this size while maintaining aspect ratio.
                         Set to None to disable rescaling. Default: 512.
        surface: Optional surface image to extract corresponding crops from.
                 If resolution differs from input, the input is resized to match
                 the surface resolution (surface is treated as ground truth).

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

    # Detect colors
    colors, quantized = detect_colors(
        img,
        max_colors=max_colors,
        min_pixel_fraction=min_area_fraction,
        background_color=background_color,
    )

    if quantize:
        img = quantize_to_colors(img, colors, background_color)

    # Save quantized image for reference
    cv2.imwrite(str(output_dir / 'quantized.png'), img)

    # --- Export full-resolution masks ---
    # 01: foreground mask (white for any non-background pixel)
    bg = np.array(background_color, dtype=np.uint8)
    foreground_mask = np.any(img != bg, axis=2).astype(np.uint8) * 255
    cv2.imwrite(str(output_dir / '01_foreground_mask.jpg'), foreground_mask)

    # 02+: per-color masks
    # Use tolerance=0 after quantization, otherwise 10
    mask_tolerance = 0 if quantize else 10
    for i, color in enumerate(colors):
        color_mask = extract_color_mask(img, color, tolerance=mask_tolerance)
        filename = f"{i+2:02d}_region_{i+1:02d}_mask.jpg"
        cv2.imwrite(str(output_dir / filename), color_mask)

    # After quantization, pixels should be exact matches (tolerance=0)
    # Without quantization, allow some tolerance for compression artifacts
    color_tolerance = 0 if quantize else 10

    # Morphological kernel for cleaning thin artifacts
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    all_regions = []

    # Process each color
    for color_idx, color in enumerate(colors):
        # Extract mask for this color
        mask = extract_color_mask(img, color, tolerance=color_tolerance)

        # Clean up thin edge artifacts with morphological opening
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)

        # Find bounding box of all pixels of this color
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            continue

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area_pixels = int(np.sum(mask > 0))

        if area_pixels < min_area_pixels:
            continue

        # Round source crop dimensions down to multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        if width < 8 or height < 8:
            continue

        # Crop the mask using the rounded dimensions
        crop = mask[y_min:y_min+height, x_min:x_min+width]

        # Compute target output dimensions (used for both mask and surface crops)
        out_w, out_h = width, height
        if output_min_size is not None:
            min_dim = min(width, height)
            if min_dim > 0:
                scale = output_min_size / min_dim
                out_w = int(round(width * scale))
                out_h = int(round(height * scale))

        # Ensure dimensions are multiples of 8 (required by many image models)
        out_w = max(8, (out_w // 8) * 8)
        out_h = max(8, (out_h // 8) * 8)

        # Rescale mask crop to target dimensions
        if (out_w, out_h) != (width, height):
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            # Re-threshold to keep it binary after interpolation
            _, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)

        # Generate filename with color info
        color_hex = f"{color[2]:02x}{color[1]:02x}{color[0]:02x}"  # BGR to RGB hex
        filename = f"region_{color_idx+1:03d}_color_{color_hex}.png"

        # Save crop
        cv2.imwrite(str(output_dir / filename), crop)

        # Save corresponding surface crop if surface image provided
        surface_filename = None
        if surface_img is not None:
            # Crop the same region from surface image (using rounded dimensions)
            surface_crop = surface_img[y_min:y_min+height, x_min:x_min+width].copy()

            # Rescale to exactly match the mask crop dimensions
            if (out_w, out_h) != (width, height):
                surface_crop = cv2.resize(surface_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            surface_filename = f"region_{color_idx+1:03d}_color_{color_hex}_surface.png"
            cv2.imwrite(str(output_dir / surface_filename), surface_crop)

        # Build metadata - minimal info needed for reconstruction
        region_info = {
            'filename': filename,
            'source_box': {
                'x': int(x_min),
                'y': int(y_min),
                'width': int(width),
                'height': int(height),
            },
        }
        if surface_filename is not None:
            region_info['surface_filename'] = surface_filename
        all_regions.append(region_info)

    # Save metadata - minimal info needed for reconstruction
    metadata = {
        'source_resolution': [original_shape[1], original_shape[0]],  # [width, height]
        'regions': all_regions,
    }

    json_path = output_dir / 'regions.json'
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
        metadata_path: Path to regions.json
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
    parser.add_argument('--max-colors', type=int, default=5,
                        help='Maximum colors to detect (default: 5)')
    parser.add_argument('--min-area', type=float, default=0.001,
                        help='Minimum region area as fraction of image (default: 0.001)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Skip quantization step')
    parser.add_argument('--output-min-size', type=int, default=512,
                        help='Rescale crops so min(w,h)=this value (default: 512, use 0 to disable)')
    parser.add_argument('--surface', help='Optional surface image to extract corresponding crops from')

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
    )

    print(f"Extracted {len(regions)} color regions to {out_path}")
    for region in regions:
        box = region['source_box']
        surface_info = f" + {region['surface_filename']}" if 'surface_filename' in region else ""
        print(f"  - {region['filename']}{surface_info}: source_box ({box['x']}, {box['y']}, {box['width']}x{box['height']})")


if __name__ == '__main__':
    _cli()
