from PIL import Image, ImageFilter
import numpy as np
import argparse
import os
import yaml

from utils import (
    align_to_reference,
    print_alignment_info,
)

# === LOAD CONFIG ===
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

subtract_config = config['subtract']
resolution_config = config['resolution']

# === INPUT ===
parser = argparse.ArgumentParser(description='Subtract background from generated keyframe to isolate creature')
parser.add_argument('generated_frame', help='Path to the AI-generated keyframe (with creature)')
parser.add_argument('background_frame', help='Path to the original darkened background frame')
parser.add_argument('--threshold', type=int, default=None,
                    help=f'Difference threshold (0-255). Default from config: {subtract_config["threshold"]}')
parser.add_argument('--boost', type=float, default=None,
                    help=f'Brightness boost multiplier. Default from config: {subtract_config["boost"]}')
parser.add_argument('--feather', type=int, default=None,
                    help=f'Edge feather radius in pixels. Default from config: {subtract_config["feather_radius"]}')
parser.add_argument('--mode', type=str, choices=['rgb', 'lab', 'luminance'], default=None,
                    help=f'Difference calculation mode. Default from config: {subtract_config["diff_mode"]}')
parser.add_argument('--output', type=str, choices=['additive', 'alpha'], default=None,
                    help=f'Output format. Default from config: {subtract_config["output_mode"]}')
parser.add_argument('--preview', action='store_true',
                    help='Also save a preview composite showing the result over the background')
parser.add_argument('--strict', action='store_true',
                    help='Fail if generated frame has different aspect ratio (instead of auto-cropping)')
args = parser.parse_args()

# Apply config defaults for unspecified args
threshold = args.threshold if args.threshold is not None else subtract_config['threshold']
boost = args.boost if args.boost is not None else subtract_config['boost']
feather_radius = args.feather if args.feather is not None else subtract_config['feather_radius']
diff_mode = args.mode if args.mode is not None else subtract_config['diff_mode']
output_mode = args.output if args.output is not None else subtract_config['output_mode']
min_alpha = subtract_config['min_alpha']
aspect_tolerance = resolution_config['aspect_tolerance']

# === LOAD IMAGES ===
# Background is the reference - generated frame will be aligned to match it
print(f'Loading background (reference): {args.background_frame}')
background = Image.open(args.background_frame).convert('RGB')
print(f'  Reference size: {background.size[0]}x{background.size[1]}')

print(f'Loading generated frame: {args.generated_frame}')
generated = Image.open(args.generated_frame).convert('RGB')
print(f'  Original size: {generated.size[0]}x{generated.size[1]}')

# Align generated frame to match background dimensions
if generated.size != background.size:
    print(f'\nAligning generated frame to reference...')
    generated, align_info = align_to_reference(generated, background, aspect_tolerance)
    print_alignment_info(align_info)

    if args.strict and align_info['cropped']:
        raise ValueError(
            f"Aspect ratio mismatch detected (strict mode). "
            f"Generated: {align_info['original_size']}, Reference: {background.size}"
        )

gen_arr = np.array(generated, dtype=np.float32)
bg_arr = np.array(background, dtype=np.float32)

# === COMPUTE DIFFERENCE MASK ===
if diff_mode == 'rgb':
    # Per-channel difference, take max across channels
    diff = np.abs(gen_arr - bg_arr)
    diff_magnitude = np.max(diff, axis=2)

elif diff_mode == 'lab':
    # Convert to LAB for perceptually uniform difference
    from skimage import color
    gen_lab = color.rgb2lab(gen_arr / 255.0)
    bg_lab = color.rgb2lab(bg_arr / 255.0)
    # Euclidean distance in LAB space
    diff_magnitude = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))
    # Normalize to 0-255 range (LAB diff can be large)
    diff_magnitude = np.clip(diff_magnitude * 2.55, 0, 255)

elif diff_mode == 'luminance':
    # Convert to grayscale and compare luminance
    gen_lum = 0.299 * gen_arr[:,:,0] + 0.587 * gen_arr[:,:,1] + 0.114 * gen_arr[:,:,2]
    bg_lum = 0.299 * bg_arr[:,:,0] + 0.587 * bg_arr[:,:,1] + 0.114 * bg_arr[:,:,2]
    diff_magnitude = np.abs(gen_lum - bg_lum)

# === CREATE MASK ===
# Binary mask based on threshold
mask = (diff_magnitude > threshold).astype(np.float32)

# Apply feathering if requested
if feather_radius > 0:
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    mask = np.array(mask_img, dtype=np.float32) / 255.0

# Apply minimum alpha
mask = np.clip(mask, min_alpha, 1.0)

# === EXTRACT CREATURE ===
# Compute the actual difference (what was added)
# For bright creatures on dark backgrounds, this is essentially the creature itself
creature_diff = gen_arr - bg_arr

# Clip to positive values only (we want additive elements)
creature_diff = np.clip(creature_diff, 0, 255)

# Apply boost
creature_diff = creature_diff * boost

# Apply mask to isolate creature
creature = creature_diff * mask[:, :, np.newaxis]

# Clip final result
creature = np.clip(creature, 0, 255).astype(np.uint8)

# === OUTPUT ===
base, ext = os.path.splitext(args.generated_frame)
params_str = f"t{threshold}_b{boost:.1f}_f{feather_radius}_{diff_mode}"

if output_mode == 'additive':
    # RGB with black background for additive blending
    output_path = f"{base}_creature_{params_str}.png"
    out_img = Image.fromarray(creature)
    out_img.save(output_path)
    print(f'Saved additive creature: {output_path}')

else:  # alpha mode
    # RGBA with transparency
    output_path = f"{base}_creature_{params_str}_alpha.png"
    alpha_channel = (mask * 255).astype(np.uint8)
    rgba = np.dstack([creature, alpha_channel])
    out_img = Image.fromarray(rgba, mode='RGBA')
    out_img.save(output_path)
    print(f'Saved alpha creature: {output_path}')

# === OPTIONAL PREVIEW ===
if args.preview:
    preview_path = f"{base}_preview_{params_str}.png"

    # Composite creature over background using additive blending
    bg_float = bg_arr.astype(np.float32)
    creature_float = creature.astype(np.float32)

    # Additive blend (screen-like effect)
    composite = bg_float + creature_float
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    preview_img = Image.fromarray(composite)
    preview_img.save(preview_path)
    print(f'Saved preview composite: {preview_path}')

# === STATS ===
mask_coverage = np.mean(mask > 0.5) * 100
creature_max = creature.max()
creature_mean = creature[creature > 0].mean() if np.any(creature > 0) else 0

print(f'\nStats:')
print(f'  Mask coverage: {mask_coverage:.1f}% of image')
print(f'  Creature max brightness: {creature_max}')
print(f'  Creature mean brightness (non-zero): {creature_mean:.1f}')
print(f'\nParameters used:')
print(f'  threshold={threshold}, boost={boost}, feather={feather_radius}, mode={diff_mode}')
