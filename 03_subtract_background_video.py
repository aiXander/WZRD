from PIL import Image, ImageFilter
import numpy as np
import argparse
import os
import yaml
import subprocess
import tempfile
import shutil
from pathlib import Path

from utils import (
    align_to_reference,
    get_aspect_ratio,
    aspect_ratios_match,
    print_alignment_info,
)

# === LOAD CONFIG ===
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

subtract_config = config['subtract']
resolution_config = config['resolution']

# === INPUT ===
parser = argparse.ArgumentParser(description='Subtract background from video to isolate creature across all frames')
parser.add_argument('input_video', help='Path to the input video (mp4)')
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
                    help='Also save a preview video showing the result composited over the background')
parser.add_argument('--crf', type=int, default=18,
                    help='FFmpeg CRF quality (0-51, lower=better). Default: 18')
parser.add_argument('--codec', type=str, default='libx264',
                    help='Video codec to use. Default: libx264')
parser.add_argument('--strict', action='store_true',
                    help='Fail if video has different aspect ratio (instead of auto-cropping)')
args = parser.parse_args()

# Apply config defaults for unspecified args
threshold = args.threshold if args.threshold is not None else subtract_config['threshold']
boost = args.boost if args.boost is not None else subtract_config['boost']
feather_radius = args.feather if args.feather is not None else subtract_config['feather_radius']
diff_mode = args.mode if args.mode is not None else subtract_config['diff_mode']
output_mode = args.output if args.output is not None else subtract_config['output_mode']
min_alpha = subtract_config['min_alpha']
aspect_tolerance = resolution_config['aspect_tolerance']


def get_video_info(video_path):
    """Get video metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split(',')
    width, height = int(parts[0]), int(parts[1])
    fps_num, fps_den = map(int, parts[2].split('/'))
    fps = fps_num / fps_den
    # nb_frames might be N/A for some formats
    try:
        frame_count = int(parts[3])
    except (ValueError, IndexError):
        frame_count = None
    return width, height, fps, frame_count


def process_frame(gen_arr, bg_arr, threshold, boost, feather_radius, diff_mode, min_alpha):
    """Process a single frame using background subtraction."""
    # === COMPUTE DIFFERENCE MASK ===
    if diff_mode == 'rgb':
        diff = np.abs(gen_arr - bg_arr)
        diff_magnitude = np.max(diff, axis=2)

    elif diff_mode == 'lab':
        from skimage import color
        gen_lab = color.rgb2lab(gen_arr / 255.0)
        bg_lab = color.rgb2lab(bg_arr / 255.0)
        diff_magnitude = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))
        diff_magnitude = np.clip(diff_magnitude * 2.55, 0, 255)

    elif diff_mode == 'luminance':
        gen_lum = 0.299 * gen_arr[:,:,0] + 0.587 * gen_arr[:,:,1] + 0.114 * gen_arr[:,:,2]
        bg_lum = 0.299 * bg_arr[:,:,0] + 0.587 * bg_arr[:,:,1] + 0.114 * bg_arr[:,:,2]
        diff_magnitude = np.abs(gen_lum - bg_lum)

    # === CREATE MASK ===
    mask = (diff_magnitude > threshold).astype(np.float32)

    if feather_radius > 0:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        mask = np.array(mask_img, dtype=np.float32) / 255.0

    mask = np.clip(mask, min_alpha, 1.0)

    # === EXTRACT CREATURE ===
    creature_diff = gen_arr - bg_arr
    creature_diff = np.clip(creature_diff, 0, 255)
    creature_diff = creature_diff * boost
    creature = creature_diff * mask[:, :, np.newaxis]
    creature = np.clip(creature, 0, 255).astype(np.uint8)

    return creature, mask


# === LOAD BACKGROUND ===
# Background is the reference - video frames will be aligned to match it
print(f'Loading background frame (reference): {args.background_frame}')
background = Image.open(args.background_frame).convert('RGB')
bg_size = background.size
bg_aspect = get_aspect_ratio(bg_size)
print(f'  Reference size: {bg_size[0]}x{bg_size[1]} (aspect: {bg_aspect:.4f})')
bg_arr = np.array(background, dtype=np.float32)

# === GET VIDEO INFO ===
print(f'\nAnalyzing video: {args.input_video}')
width, height, fps, frame_count = get_video_info(args.input_video)
video_aspect = width / height
print(f'  Resolution: {width}x{height} (aspect: {video_aspect:.4f})')
print(f'  Frame rate: {fps:.2f} fps')
if frame_count:
    print(f'  Frame count: {frame_count}')

# Check for resolution/aspect ratio mismatch
needs_alignment = (width, height) != bg_size
aspect_mismatch = not aspect_ratios_match(video_aspect, bg_aspect, aspect_tolerance)

if needs_alignment:
    print(f'\n*** Resolution mismatch detected ***')
    print(f'  Video: {width}x{height}')
    print(f'  Background: {bg_size[0]}x{bg_size[1]}')

    if aspect_mismatch:
        print(f'  Aspect ratio mismatch: video={video_aspect:.4f}, bg={bg_aspect:.4f}')
        if args.strict:
            raise ValueError(
                f"Aspect ratio mismatch (strict mode). "
                f"Video: {width}x{height} ({video_aspect:.4f}), "
                f"Background: {bg_size[0]}x{bg_size[1]} ({bg_aspect:.4f})"
            )
        print(f'  Will center-crop frames to match background aspect ratio')
    else:
        print(f'  Will resize frames to match background resolution')
else:
    print(f'  Resolution matches background - no alignment needed')

# === SETUP OUTPUT PATHS ===
base, ext = os.path.splitext(args.input_video)
params_str = f"t{threshold}_b{boost:.1f}_f{feather_radius}_{diff_mode}"

if output_mode == 'alpha':
    output_path = f"{base}_creature_{params_str}_alpha.mov"  # MOV supports alpha
    pix_fmt = 'rgba'
else:
    output_path = f"{base}_creature_{params_str}.mp4"
    pix_fmt = 'yuv420p'

preview_path = f"{base}_preview_{params_str}.mp4" if args.preview else None

# === CREATE TEMP DIRECTORY FOR FRAMES ===
temp_dir = tempfile.mkdtemp(prefix='bg_subtract_')
temp_frames_dir = os.path.join(temp_dir, 'frames')
temp_output_dir = os.path.join(temp_dir, 'output')
temp_preview_dir = os.path.join(temp_dir, 'preview') if args.preview else None

os.makedirs(temp_frames_dir)
os.makedirs(temp_output_dir)
if temp_preview_dir:
    os.makedirs(temp_preview_dir)

try:
    # === EXTRACT FRAMES ===
    print(f'\nExtracting frames...')
    extract_cmd = [
        'ffmpeg', '-i', args.input_video,
        '-vsync', '0',
        os.path.join(temp_frames_dir, 'frame_%06d.png')
    ]
    subprocess.run(extract_cmd, capture_output=True, check=True)

    frame_files = sorted(Path(temp_frames_dir).glob('frame_*.png'))
    total_frames = len(frame_files)
    print(f'  Extracted {total_frames} frames')

    # === PROCESS FRAMES ===
    print(f'\nProcessing frames...')
    print(f'  Parameters: threshold={threshold}, boost={boost}, feather={feather_radius}, mode={diff_mode}')
    if needs_alignment:
        print(f'  Frame alignment: enabled (target: {bg_size[0]}x{bg_size[1]})')

    alignment_logged = False  # Only log alignment details for first frame

    for i, frame_path in enumerate(frame_files):
        if (i + 1) % 30 == 0 or i == 0 or i == total_frames - 1:
            print(f'  Processing frame {i + 1}/{total_frames}')

        # Load frame
        frame = Image.open(frame_path).convert('RGB')

        # Align frame to background if needed
        if needs_alignment:
            frame, align_info = align_to_reference(frame, background, aspect_tolerance)
            if not alignment_logged:
                print_alignment_info(align_info, prefix='    ')
                alignment_logged = True

        gen_arr = np.array(frame, dtype=np.float32)

        # Process
        creature, mask = process_frame(gen_arr, bg_arr, threshold, boost, feather_radius, diff_mode, min_alpha)

        # Save output frame
        frame_name = frame_path.name
        if output_mode == 'alpha':
            alpha_channel = (mask * 255).astype(np.uint8)
            rgba = np.dstack([creature, alpha_channel])
            out_img = Image.fromarray(rgba, mode='RGBA')
        else:
            out_img = Image.fromarray(creature)

        out_img.save(os.path.join(temp_output_dir, frame_name))

        # Save preview frame if requested
        if args.preview:
            bg_float = bg_arr.astype(np.float32)
            creature_float = creature.astype(np.float32)
            composite = np.clip(bg_float + creature_float, 0, 255).astype(np.uint8)
            preview_img = Image.fromarray(composite)
            preview_img.save(os.path.join(temp_preview_dir, frame_name))

    # === ENCODE OUTPUT VIDEO ===
    print(f'\nEncoding output video...')

    if output_mode == 'alpha':
        # Use ProRes 4444 for alpha channel support
        encode_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_output_dir, 'frame_%06d.png'),
            '-c:v', 'prores_ks',
            '-profile:v', '4444',
            '-pix_fmt', 'yuva444p10le',
            output_path
        ]
    else:
        encode_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_output_dir, 'frame_%06d.png'),
            '-c:v', args.codec,
            '-crf', str(args.crf),
            '-pix_fmt', pix_fmt,
            output_path
        ]

    subprocess.run(encode_cmd, capture_output=True, check=True)
    print(f'  Saved: {output_path}')

    # === ENCODE PREVIEW VIDEO ===
    if args.preview:
        print(f'\nEncoding preview video...')
        preview_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_preview_dir, 'frame_%06d.png'),
            '-c:v', args.codec,
            '-crf', str(args.crf),
            '-pix_fmt', 'yuv420p',
            preview_path
        ]
        subprocess.run(preview_cmd, capture_output=True, check=True)
        print(f'  Saved: {preview_path}')

finally:
    # === CLEANUP ===
    print(f'\nCleaning up temporary files...')
    shutil.rmtree(temp_dir)

# === SUMMARY ===
print(f'\n{"="*50}')
print(f'Video processing complete!')
print(f'  Input: {args.input_video}')
print(f'  Output: {output_path}')
if args.preview:
    print(f'  Preview: {preview_path}')
print(f'\nParameters used:')
print(f'  threshold={threshold}, boost={boost}, feather={feather_radius}, mode={diff_mode}')
if needs_alignment:
    print(f'\nAlignment applied:')
    print(f'  Video resolution: {width}x{height}')
    print(f'  Output resolution: {bg_size[0]}x{bg_size[1]}')
    if aspect_mismatch:
        print(f'  Frames were center-cropped to match aspect ratio')
