from PIL import Image
import numpy as np
import argparse
import os
import yaml

from utils import (
    parse_aspect_ratio,
    normalize_image,
    center_crop_to_aspect,
    resize_to_base_resolution,
    get_aspect_ratio,
    aspect_ratios_match,
    compute_target_dimensions,
)

# === LOAD CONFIG ===
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

darken_config = config['darken']
resolution_config = config['resolution']

# === INPUT ===
parser = argparse.ArgumentParser(description='Darken an image for projection mapping')
parser.add_argument('input_image', help='Path to the input image')
parser.add_argument('--gamma', type=float, default=None,
                    help=f'Dynamic range compression. Default from config: {darken_config["gamma"]}')
parser.add_argument('--max-brightness', type=float, default=None,
                    help=f'Maximum output intensity (0.0-1.0). Default from config: {darken_config["max_brightness"]}')
parser.add_argument('--plot', action='store_true', help='Save RGB histogram of the output image')
parser.add_argument('--aspect', type=str, default=None,
                    help=f'Target aspect ratio for center crop (e.g., 16:9, 4:3, 1:1). Default from config: {resolution_config["default_aspect"]}')
parser.add_argument('--base-resolution', type=int, default=None,
                    help=f'Base resolution (max dimension). Default from config: {resolution_config["base_resolution"]}')
parser.add_argument('--no-normalize', action='store_true',
                    help='Skip aspect ratio and resolution normalization (keep original dimensions)')
args = parser.parse_args()

input_path = args.input_image

# Apply config defaults for unspecified args
gamma = args.gamma if args.gamma is not None else darken_config['gamma']
max_brightness = args.max_brightness if args.max_brightness is not None else darken_config['max_brightness']
aspect_str = args.aspect if args.aspect is not None else resolution_config['default_aspect']
base_resolution = args.base_resolution if args.base_resolution is not None else resolution_config['base_resolution']

base, ext = os.path.splitext(input_path)
output_path = f"{base}_darkened_{gamma:.2f}.jpg"

# === LOAD AND NORMALIZE ===
img = Image.open(input_path).convert('RGB')
original_size = img.size
print(f'Loaded: {input_path} ({original_size[0]}x{original_size[1]})')

if not args.no_normalize:
    # Parse aspect ratio
    target_aspect = parse_aspect_ratio(aspect_str)
    target_dims = compute_target_dimensions(target_aspect, base_resolution)

    # Normalize: crop to aspect ratio, then resize to base resolution
    img = normalize_image(img, target_aspect, base_resolution)

    print(f'Normalized to {aspect_str} @ {img.size[0]}x{img.size[1]} (base resolution: {base_resolution})')

    if img.size != original_size:
        # Save the normalized (but not darkened) version as reference
        normalized_path = f"{base}_normalized_{aspect_str.replace(':', 'x')}_{img.size[0]}x{img.size[1]}{ext}"
        img.save(normalized_path, quality=95)
        print(f'Saved normalized reference: {normalized_path}')

arr = np.array(img, dtype=np.float32) / 255.0

# Apply gamma compression (logarithmic-like curve)
# This expands shadow detail while compressing highlights
arr = np.power(arr, gamma)

# Normalize to use full range, then scale to max_brightness
arr_min, arr_max = arr.min(), arr.max()
if arr_max > arr_min:
    arr = (arr - arr_min) / (arr_max - arr_min)

arr = arr * max_brightness

# Convert back to 8-bit
arr = (arr * 255).astype(np.uint8)

out_img = Image.fromarray(arr)
out_img.save(output_path)

print(f'Saved {output_path} (max_brightness={max_brightness}, gamma={gamma})')

# === HISTOGRAM ===
if args.plot:
    import matplotlib.pyplot as plt

    histogram_path = f"{base}_darkened_hist_{gamma:.2f}.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']

    for i, (color, label) in enumerate(zip(colors, labels)):
        channel = arr[:, :, i].flatten()
        ax.hist(channel, bins=256, range=(0, 255), alpha=0.5, color=color, label=label)

    ax.set_xlabel('Pixel Intensity (0-255)')
    ax.set_ylabel('Frequency')
    ax.set_title('RGB Intensity Distribution')
    ax.legend()
    max_v = arr.max()
    ax.set_xlim(0, min(max_v * 1.2, 255))

    plt.tight_layout()
    plt.savefig(histogram_path, dpi=150)
    plt.close()

    print(f'Saved histogram: {histogram_path}')
