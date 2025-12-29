"""
Image darkening for projection mapping.

Functions for reducing image brightness while preserving detail,
preparing surfaces for projection mapping applications.
"""

from PIL import Image
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path

from .utils import (
    parse_aspect_ratio,
    normalize_image,
    compute_target_dimensions,
)
from .config import get_darken_config, get_resolution_config


def darken_image(
    image: Union[Image.Image, np.ndarray],
    gamma: float = 1.5,
    max_brightness: float = 0.15,
) -> np.ndarray:
    """
    Darken an image for projection mapping.

    Applies gamma compression to expand shadow detail while compressing highlights,
    then scales to the specified maximum brightness.

    Args:
        image: PIL Image or numpy array (HWC, 0-255)
        gamma: Dynamic range compression factor. Lower = more compression.
        max_brightness: Maximum output intensity (0.0-1.0)

    Returns:
        Darkened image as numpy array (uint8, 0-255)
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    else:
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    # Apply gamma compression (logarithmic-like curve)
    # This expands shadow detail while compressing highlights
    arr = np.power(arr, gamma)

    # Normalize to use full range, then scale to max_brightness
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)

    arr = arr * max_brightness

    # Convert back to 8-bit
    return (arr * 255).astype(np.uint8)


def darken_image_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    gamma: Optional[float] = None,
    max_brightness: Optional[float] = None,
    target_aspect: Optional[str] = None,
    base_resolution: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[Image.Image, dict]:
    """
    Load, optionally normalize, and darken an image file.

    Args:
        input_path: Path to input image
        output_path: Path to save output (optional)
        gamma: Dynamic range compression (default from config)
        max_brightness: Maximum brightness 0.0-1.0 (default from config)
        target_aspect: Aspect ratio string e.g. "16:9" (default from config)
        base_resolution: Max dimension in pixels (default from config)
        normalize: Whether to normalize aspect ratio and resolution

    Returns:
        Tuple of (darkened PIL Image, info dict with processing details)
    """
    input_path = Path(input_path)

    # Load config defaults
    darken_config = get_darken_config()
    resolution_config = get_resolution_config()

    gamma = gamma if gamma is not None else darken_config['gamma']
    max_brightness = max_brightness if max_brightness is not None else darken_config['max_brightness']
    target_aspect_str = target_aspect if target_aspect is not None else resolution_config['default_aspect']
    base_res = base_resolution if base_resolution is not None else resolution_config['base_resolution']

    # Load image
    img = Image.open(input_path).convert('RGB')
    original_size = img.size

    info = {
        'original_size': original_size,
        'gamma': gamma,
        'max_brightness': max_brightness,
        'normalized': False,
        'target_aspect': None,
        'final_size': None,
    }

    # Normalize if requested
    if normalize:
        target_aspect_val = parse_aspect_ratio(target_aspect_str)
        img = normalize_image(img, target_aspect_val, base_res)
        info['normalized'] = True
        info['target_aspect'] = target_aspect_str

    info['final_size'] = img.size

    # Darken
    darkened_arr = darken_image(img, gamma=gamma, max_brightness=max_brightness)
    darkened_img = Image.fromarray(darkened_arr)

    # Save if output path provided
    if output_path is not None:
        darkened_img.save(output_path, quality=95)
        info['output_path'] = str(output_path)

    return darkened_img, info
