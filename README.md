# Projection Mapping Pipeline

A workflow for creating projection-mapped animations where AI-generated characters appear to inhabit real-world surfaces. The pipeline generates content optimized for additive projection by isolating animated elements from static backgrounds.

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

### Step 1: Capture Projection Surface

Photograph the physical surface where you will project (building facade, wall, sculpture, etc.). This image serves as the spatial reference for all generated content.

**Requirements:**
- High resolution capture
- Stable camera position (matching your projector's perspective)
- Ambient lighting conditions similar to projection time

### Step 2: Darken the Background Image

**Script:** `01_darken_image.py`

Transforms the surface photograph into a subtle, darkened guidance image. This becomes the static background that AI models will use as context for generating characters.

```bash
python 01_darken_image.py input_photo.jpg --gamma 1.5 --max-brightness 0.15
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gamma` | 1.5 | Dynamic range compression. Lower = more shadow detail |
| `--max-brightness` | 0.15 | Maximum output intensity (0.0-1.0). 0.15 = 15% of full white |
| `--aspect` | None | Target aspect ratio for center crop (e.g., 16:9) |
| `--plot` | False | Save RGB histogram of output |

**Output:** `{input}_darkened_{gamma}.jpg`

### Step 3: Generate AI Keyframes

Use an AI image generator (DALL-E, Midjourney, Stable Diffusion, etc.) to create keyframes featuring your character at various positions on the darkened background.

**Key requirements for prompting:**
- Keep the background fixed and nearly invisible
- Position character at different locations for animation
- Maintain exact aspect ratio and framing
- Character should emit/reflect light onto immediate surroundings

Example prompt structure:
```
Generate an image of [character description] positioned at [location] on this building.
Keep the building incredibly dark and faint. Let the character illuminate its
immediate surroundings. The output must have exactly the same aspect ratio and
framing as the input—they must match 1-to-1 for overlay.
```

### Step 4: Generate Video Transitions

Use an AI img2img video generator to create smooth transitions between keyframes.

**Requirements:**
- Fixed camera (no panning, zooming, or movement)
- Static background throughout
- Only the character should animate
- Consistent lighting and style across transitions

### Step 5: Subtract Background from Video

**Script:** `03_subtract_background_video.py`

Processes the AI-generated video to isolate only the animated character, removing all unchanged background regions. This creates projection-ready content where black pixels = no light projected.

```bash
python 03_subtract_background_video.py video.mp4 background_darkened.jpg
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 10 | Difference threshold (0-255). Higher = more aggressive background removal |
| `--boost` | 1.1 | Brightness multiplier for extracted content |
| `--feather` | 4 | Edge feather radius in pixels for soft edges |
| `--mode` | luminance | Difference calculation: `rgb`, `lab`, or `luminance` |
| `--output` | additive | Output format: `additive` (black bg) or `alpha` (transparency) |
| `--preview` | False | Also save preview composite |
| `--crf` | 18 | FFmpeg quality (0-51, lower = better) |

**Output:** `{input}_creature_{params}.mp4` (or `.mov` for alpha)

### Step 6: Project

Play the processed video through your projector, aligned to the physical surface. The additive blending means:
- Black pixels (unchanged regions) → No light projected
- Bright pixels (character/changes) → Light projected

## Single Frame Processing

For testing or still images, use `02_subtract_background.py`:

```bash
python 02_subtract_background.py keyframe.png background_darkened.jpg --preview
```

## Resolution Handling

The pipeline is designed to handle resolution and aspect ratio mismatches that commonly occur when using different AI tools.

### How It Works

1. **Step 2 (Darken)** establishes the canonical resolution:
   - Crops input to target aspect ratio (default: 16:9)
   - Resizes to base resolution (default: max dimension = 1920px)
   - This becomes the reference that all subsequent steps align to

2. **Steps 3-4 (AI Generation)** may produce slightly different resolutions:
   - AI image generators often output at their native resolutions
   - AI video generators may change dimensions slightly

3. **Step 5 (Subtraction)** automatically aligns frames:
   - Detects resolution mismatches between video and background
   - Center-crops frames if aspect ratios differ
   - Resizes frames to match background dimensions
   - Ensures consistent subtraction across all frames

### Strict Mode

Use `--strict` to fail on aspect ratio mismatches instead of auto-cropping:

```bash
python 03_subtract_background_video.py video.mp4 bg.jpg --strict
```

This is useful when you need to verify AI tools are respecting your aspect ratio.

## Configuration

All default parameters are stored in `config.yaml`:

```yaml
resolution:
  default_aspect: "16:9"      # Default aspect ratio for normalization
  base_resolution: 1920       # Max dimension in pixels
  aspect_tolerance: 0.02      # 2% tolerance for aspect matching

darken:
  max_brightness: 0.15
  gamma: 1.5

subtract:
  threshold: 10
  boost: 1.1
  feather_radius: 4
  diff_mode: "luminance"
  min_alpha: 0.0
  output_mode: "additive"
```

## Dependencies

- Python 3.8+
- PIL/Pillow
- NumPy
- PyYAML
- FFmpeg (for video processing)
- scikit-image (optional, for LAB color mode)
- matplotlib (optional, for histogram plots)

Install Python dependencies:
```bash
pip install pillow numpy pyyaml scikit-image matplotlib
```

## File Structure

```
├── config.yaml                       # Pipeline configuration
├── utils.py                          # Resolution/alignment utilities
├── 01_darken_image.py               # Step 2: Darken background
├── 02_subtract_background.py        # Single-frame background subtraction
├── 03_subtract_background_video.py  # Step 5: Video background subtraction
└── pipeline.py                      # Prompt templates/examples
```

## Tips

- **Threshold tuning:** Start low (5-10) and increase if background artifacts appear
- **Boost values:** Use 1.0-1.5 for natural brightness, higher for more dramatic effect
- **Feathering:** Higher values (8-16) give softer edges but may lose fine detail
- **Mode selection:**
  - `luminance` - Fast, good for bright characters on dark backgrounds
  - `lab` - More perceptually accurate, catches subtle color differences
  - `rgb` - Simple per-channel difference
