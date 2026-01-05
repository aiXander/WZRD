"""
WZRD - VJ Projection Mapping Utilities

A toolkit for processing images and videos for projection mapping,
including image darkening, background subtraction for creature extraction,
and island segmentation.

Example usage:
    from wzrd import darken_image, subtract_background

    # Darken an image for projection
    darkened = darken_image(my_image, gamma=1.5, max_brightness=0.15)

    # Extract creature from generated frame
    creature, mask, info = subtract_background(generated, background)
"""

__version__ = "0.1.0"

# Core image processing
from .darken import (
    darken_image,
    darken_image_file,
)

# Background subtraction
from .subtract import (
    subtract_background,
    subtract_background_file,
    compute_difference_mask,
    extract_creature,
)

# Video processing
from .video import (
    subtract_background_video,
    get_video_info,
    iter_video_frames,
    process_frame,
)

# Island extraction
from .islands import (
    find_islands,
    crop_island,
    crop_islands,
    save_islands,
    crop_islands_file,
)

# Reprojection (islands back to full canvas)
from .reproject import (
    reproject_video,
    reproject_videos_batch,
    reproject_from_islands_dir,
    reproject_video_with_aspect,
    load_island_metadata,
)

# Utilities
from .utils import (
    # Aspect ratio handling
    parse_aspect_ratio,
    get_aspect_ratio,
    aspect_ratios_match,
    get_closest_standard_aspect,
    compute_target_dimensions,
    ASPECT_RATIOS,
    # Image operations
    center_crop_to_aspect,
    resize_to_base_resolution,
    normalize_image,
    align_to_reference,
    align_array_to_reference,
    load_and_normalize,
    load_and_align,
    # Constants
    DEFAULT_BASE_RESOLUTION,
    ASPECT_RATIO_TOLERANCE,
)

# Configuration
from .config import (
    load_config,
    get_darken_config,
    get_subtract_config,
    get_resolution_config,
    DEFAULT_CONFIG,
)

__all__ = [
    # Version
    '__version__',
    # Darken
    'darken_image',
    'darken_image_file',
    # Subtract
    'subtract_background',
    'subtract_background_file',
    'compute_difference_mask',
    'extract_creature',
    # Video
    'subtract_background_video',
    'get_video_info',
    'iter_video_frames',
    'process_frame',
    # Islands
    'find_islands',
    'crop_island',
    'crop_islands',
    'save_islands',
    'crop_islands_file',
    # Reproject
    'reproject_video',
    'reproject_videos_batch',
    'reproject_from_islands_dir',
    'reproject_video_with_aspect',
    'load_island_metadata',
    # Utils
    'parse_aspect_ratio',
    'get_aspect_ratio',
    'aspect_ratios_match',
    'get_closest_standard_aspect',
    'compute_target_dimensions',
    'ASPECT_RATIOS',
    'center_crop_to_aspect',
    'resize_to_base_resolution',
    'normalize_image',
    'align_to_reference',
    'align_array_to_reference',
    'load_and_normalize',
    'load_and_align',
    'DEFAULT_BASE_RESOLUTION',
    'ASPECT_RATIO_TOLERANCE',
    # Config
    'load_config',
    'get_darken_config',
    'get_subtract_config',
    'get_resolution_config',
    'DEFAULT_CONFIG',
]
