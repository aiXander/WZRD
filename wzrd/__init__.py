"""
WZRD - VJ Projection Mapping Utilities

A toolkit for processing images and videos for projection mapping,
including image darkening, background subtraction for creature extraction,
island segmentation, and projection surface detection.

Example usage:
    from wzrd import darken_image, subtract_background, detect_projection_area, align_images

    # Darken an image for projection (gradient-weighted histogram equalization)
    darkened = darken_image(my_image, max_brightness=0.20)

    # Extract creature from generated frame
    creature, mask, info = subtract_background(generated, background)

    # Detect projection surface from a photo
    cropped, info = detect_projection_area("photo_of_projection.jpg", margin=0.02)

    # Align a well-lit photo to match a projection crop
    aligned, info = align_images(lit_surface, target_crop)
"""

__version__ = "0.1.0"

# Background subtraction (always available)
from .subtract_frame import (
    subtract_background,
    subtract_background_file,
)

# Core image-processing primitives (always available via utils)
from .utils import (
    compute_difference_mask,
    extract_creature,
    correct_color_shift,
    guided_filter,
    rgb_to_lab,
    lab_to_rgb,
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

__all__ = [
    # Version
    '__version__',
    # Darken (lazy)
    'darken_image',
    'darken_image_file',
    # Subtract
    'subtract_background',
    'subtract_background_file',
    'compute_difference_mask',
    'extract_creature',
    'correct_color_shift',
    'guided_filter',
    'rgb_to_lab',
    'lab_to_rgb',
    # Video (lazy)
    'subtract_background_video',
    'get_video_info',
    'iter_video_frames',
    'process_frame',
    # Islands (lazy)
    'find_islands',
    'crop_island',
    'crop_islands',
    'save_islands',
    'crop_islands_file',
    'extract_color_regions',
    'reassemble_regions',
    # Reproject (lazy)
    'reproject_video',
    'reproject_videos_batch',
    'reproject_from_islands_dir',
    'reproject_video_with_aspect',
    'load_island_metadata',
    # Projection detection (lazy)
    'detect_projection_area',
    'detect_projection_from_array',
    # Alignment (lazy)
    'align_images',
    'align_images_file',
    'AlignmentResult',
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
]


# Lazy imports for modules that may be missing or are run as __main__
def __getattr__(name):
    if name in ('darken_image', 'darken_image_file'):
        from . import darken
        return getattr(darken, name)
    if name in ('align_images', 'align_images_file', 'AlignmentResult'):
        from . import align
        return getattr(align, name)
    if name in ('find_islands', 'crop_island', 'crop_islands', 'save_islands',
                'crop_islands_file', 'extract_color_regions', 'reassemble_regions'):
        from . import islands
        return getattr(islands, name)
    if name in ('subtract_background_video', 'get_video_info',
                'iter_video_frames', 'process_frame'):
        from . import subtract_video
        return getattr(subtract_video, name)
    if name in ('reproject_video', 'reproject_videos_batch',
                'reproject_from_islands_dir', 'reproject_video_with_aspect',
                'load_island_metadata'):
        from . import reproject
        return getattr(reproject, name)
    if name in ('detect_projection_area', 'detect_projection_from_array'):
        from . import detect
        return getattr(detect, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
