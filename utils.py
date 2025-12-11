"""
Utility functions for projection mapping pipeline.

Handles resolution normalization, aspect ratio cropping, and image alignment
to ensure consistency across AI-generated content.
"""

from PIL import Image
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path


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

    Args:
        image: PIL Image to normalize
        target_aspect: Target aspect ratio (width/height)
        base_resolution: Target max dimension

    Returns:
        Normalized PIL Image
    """
    # First crop to aspect ratio
    cropped = center_crop_to_aspect(image, target_aspect)

    # Then resize to base resolution
    resized = resize_to_base_resolution(cropped, base_resolution)

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
        height = int(base_resolution / target_aspect)
    else:
        # Portrait: height is the max dimension
        height = base_resolution
        width = int(base_resolution * target_aspect)

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
