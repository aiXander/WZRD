# WZRD - Projection Mapping Toolkit

A Python toolkit for additive projection mapping. Extracts animated elements from static backgrounds so only the changing regions get projected — unchanged areas stay invisible, making characters appear to inhabit real surfaces.

## Installation

```bash
pip install -e .

# With island extraction support (requires scikit-learn for KMeans)
pip install -e ".[islands]"

# From GitHub
pip install "wzrd @ git+https://github.com/xandersteenbrugge/WZRD.git"
```

**Requirements:** Python 3.10+, FFmpeg (for video processing)

## Modules

Every module runs standalone via `python -m wzrd.<module>`. All processing parameters have sensible defaults.

### detect — Projection Surface Detection

Automatically finds the projection area in a photo of the lit surface. Uses color proximity mapping, Otsu thresholding, and convex hull fitting.

```bash
python -m wzrd.detect photo.jpg -o cropped.png --margin 0.02 --aspect-ratio 16:9
```

### darken — Gradient-Weighted Darkening

Darkens well-lit surface photos into faint guidance images for AI generation. Uses gradient-weighted histogram equalization in CIELAB space to preserve texture detail while reducing overall brightness.

```bash
python -m wzrd.darken surface.jpg -o dark.png --max-brightness 0.25 --aspect 16:9
```

### align — Image Alignment

Aligns a source image to a target via feature matching (SIFT/AKAZE + RANSAC) with template matching fallback and ECC sub-pixel refinement.

```bash
python -m wzrd.align source.png target.png -o aligned.png
```

### subtract_frame — Single-Frame Background Subtraction

Isolates the "creature" (changed regions) from a generated frame against its background. Pipeline: color correction → LAB difference → soft ramp threshold → morphological cleanup → guided filter feathering → gamma-corrected extraction.

```bash
python -m wzrd.subtract_frame generated.png background.png -o creature.png --preview
```

### subtract_video — Video Background Subtraction

Processes entire videos frame-by-frame with multi-resolution acceleration and temporal EMA smoothing. Color correction runs at 1/4 res, masks at 1/2 res, guided filter at full res.

```bash
python -m wzrd.subtract_video animation.mp4 background.png -o output.mp4 --temporal-smoothing 0.3
```

### islands — Color Region Extraction

Segments images into distinct color regions via K-means clustering, extracts connected components, and saves per-region crops with JSON metadata for reconstruction.

```bash
python -m wzrd.islands input.png -o regions/ --max-colors 6 --surface surface.png
```

### reproject — Layer Compositing

Reprojects cropped island videos back onto a full-size canvas at their original positions, for layer-based compositing in tools like Resolume.

```bash
python -m wzrd.reproject regions_dir/ -o output.mp4 --aspect 16:9
```

## Python API

```python
import wzrd

# Background subtraction (single frame)
creature, mask, info = wzrd.subtract_background(generated, background,
    threshold=10, ramp=20, gamma=0.85, diff_mode='lab')

# Background subtraction (video)
wzrd.subtract_background_video("video.mp4", "background.png",
    output_path="output.mp4", temporal_smoothing=0.3)

# Darken image
from wzrd import darken_image
darkened = darken_image(image, max_brightness=0.25)

# Detect projection surface
from wzrd import detect_projection_area
detect_projection_area("photo.jpg", margin=0.02, target_aspect_ratio=16/9)

# Align images
from wzrd import align_images
warped, result = align_images(source, target)

# Extract color regions
from wzrd import extract_color_regions
extract_color_regions(image, output_dir="regions/", max_colors=6)

# Reproject layers
from wzrd import reproject_videos_batch
reproject_videos_batch(video_json_pairs, canvas_width=1920, canvas_height=1080)
```

## Dependencies

- **Pillow**, **NumPy**, **OpenCV** (core)
- **FFmpeg** (system install, for video I/O)
- **scikit-learn** (optional, for `islands` color clustering)

## Tips

- **Threshold tuning:** Start low (5–10) and increase if background bleeds through
- **Diff mode:** `lab` is most perceptually accurate; `luminance` is faster; `rgb` is simplest
- **Temporal smoothing:** 0.3 gives moderate stability; 0 disables for per-frame independence
- **Gamma < 1.0** brightens extracted creatures (default 0.85), useful for projection
- **Feathering:** Higher radius gives softer edges but costs performance in video mode
