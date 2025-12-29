"""
Island extraction from binary segmentation maps.

Functions to find and crop connected components (islands) from
binary images, useful for extracting individual projection regions.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Literal


Connectivity = Literal[4, 8]


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
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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
