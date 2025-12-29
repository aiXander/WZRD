# WZRD - Projection Mapping Pipeline

A Python toolkit for creating projection-mapped animations where AI-generated characters appear to inhabit real-world surfaces. The pipeline generates content optimized for additive projection by isolating animated elements from static backgrounds.

## Installation

```bash
# Install from local directory
pip install -e .

# Install with optional dependencies (LAB color mode, histogram plots)
pip install -e ".[all]"

# Install from GitHub (for use in other projects)
pip install "wzrd @ git+https://github.com/xandersteenbrugge/WZRD.git"
```

## Overview

This pipeline creates projection content where a luminous character (or any animated element) appears on a physical surface. By subtracting the static background, only the regions with visual change get projected—making unchanged areas invisible and creating the illusion that the character truly exists on the surface.

## Pipeline Steps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROJECTION MAPPING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │  1. CAPTURE  │         │  2. DARKEN   │         │ 3. GENERATE  │
   │              │  ────►  │              │  ────►  │   KEYFRAMES  │
   │ Photograph   │         │ Create faint │         │              │
   │ projection   │         │ background   │         │ AI creates   │
   │ surface      │         │ guidance     │         │ character    │
   │              │         │ image        │         │ poses        │
   └──────────────┘         └──────────────┘         └──────────────┘
                                                           │
                                                           ▼
   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │ 5. SUBTRACT  │         │ 4. GENERATE  │         │   KEYFRAME   │
   │   BACKGROUND │  ◄────  │    VIDEO     │  ◄────  │   SEQUENCE   │
   │              │         │              │         │              │
   │ Isolate only │         │ AI img2img   │         │ Multiple     │
   │ changed      │         │ video creates│         │ character    │
   │ regions      │         │ transitions  │         │ positions    │
   └──────────────┘         └──────────────┘         └──────────────┘
         │
         ▼
   ┌──────────────┐
   │  6. PROJECT  │
   │              │
   │ Additive     │
   │ projection   │
   │ onto surface │
   └──────────────┘
```

## Python API

### Darken Images

```python
from wzrd import darken_image, darken_image_file
from PIL import Image

# Darken an image in memory
img = Image.open("surface.jpg")
darkened = darken_image(img, gamma=1.5, max_brightness=0.15)

# Or process a file directly
darkened_img, info = darken_image_file(
    "surface.jpg",
    output_path="surface_dark.jpg",
    gamma=1.5,
    max_brightness=0.15,
    target_aspect="16:9",
    base_resolution=1920,
)
```

### Background Subtraction (Images)

```python
from wzrd import subtract_background, subtract_background_file

# Process images in memory
creature, mask, info = subtract_background(
    generated_frame,      # PIL Image or numpy array
    background_frame,     # PIL Image or numpy array
    threshold=10,
    boost=1.1,
    feather_radius=4,
    diff_mode='luminance',  # 'rgb', 'lab', or 'luminance'
    output_mode='additive', # 'additive' or 'alpha'
)

# Or process files
creature_img, info = subtract_background_file(
    "generated.png",
    "background.png",
    output_path="creature.png",
    preview=True,  # Also save composite preview
)
```

### Background Subtraction (Video)

```python
from wzrd import subtract_background_video

info = subtract_background_video(
    "animation.mp4",
    "background.png",
    output_path="creature_video.mp4",
    threshold=10,
    boost=1.1,
    preview=True,
    progress_callback=lambda frame, total: print(f"Frame {frame}/{total}"),
)
```

### Island Extraction

Extract connected components from binary segmentation maps:

```python
from wzrd import find_islands, crop_islands_file

# Find islands in memory
islands, labels, binary = find_islands(segmentation_image, min_area=100)

# Process file and save crops
islands, output_dir = crop_islands_file(
    "segmentation.png",
    output_dir="islands_output",
    min_area=100,
    connectivity=8,
)
```

### Utility Functions

```python
from wzrd import (
    parse_aspect_ratio,
    normalize_image,
    align_to_reference,
    center_crop_to_aspect,
)

# Parse aspect ratio strings
aspect = parse_aspect_ratio("16:9")  # Returns 1.7778

# Normalize image to aspect ratio and resolution
normalized = normalize_image(img, target_aspect=1.7778, base_resolution=1920)

# Align generated frame to match reference dimensions
aligned, info = align_to_reference(generated, reference, tolerance=0.02)
```

## CLI Commands

The package installs these command-line tools:

```bash
# Darken an image
wzrd-darken image.jpg --gamma 1.5 --max-brightness 0.15 --aspect 16:9

# Subtract background from image
wzrd-subtract generated.png background.png --threshold 10 --boost 1.1 --preview

# Subtract background from video
wzrd-subtract-video video.mp4 background.png --threshold 10 --preview

# Extract islands from segmentation map
wzrd-crop-islands segmentation.png --min-area 100
```

## Configuration

Default parameters are stored in `wzrd/default_config.yaml`. You can override by placing a `config.yaml` in your working directory:

```yaml
resolution:
  default_aspect: "16:9"
  base_resolution: 1920
  aspect_tolerance: 0.02

darken:
  max_brightness: 0.15
  gamma: 1.5

subtract:
  threshold: 10
  boost: 1.1
  feather_radius: 4
  diff_mode: "luminance"  # rgb, lab, or luminance
  min_alpha: 0.0
  output_mode: "additive"  # additive or alpha
```

Load custom config programmatically:

```python
from wzrd import load_config

config = load_config("my_config.yaml")
```

## Dependencies

**Required:**
- Python 3.10+
- Pillow
- NumPy
- OpenCV
- PyYAML
- FFmpeg (system install, for video processing)

**Optional:**
- scikit-image (for LAB color mode)
- matplotlib (for histogram plots)

## Tips

- **Threshold tuning:** Start low (5-10) and increase if background artifacts appear
- **Boost values:** Use 1.0-1.5 for natural brightness, higher for more dramatic effect
- **Feathering:** Higher values (8-16) give softer edges but may lose fine detail
- **Mode selection:**
  - `luminance` - Fast, good for bright characters on dark backgrounds
  - `lab` - More perceptually accurate, catches subtle color differences
  - `rgb` - Simple per-channel difference
