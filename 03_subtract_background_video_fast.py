from PIL import Image
import numpy as np
import argparse
import os
import yaml
import subprocess
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from utils import (
    align_to_reference,
    get_aspect_ratio,
    aspect_ratios_match,
    print_alignment_info,
)

# Pre-import skimage for LAB mode (avoid import inside loop)
try:
    from skimage import color as skimage_color
except ImportError:
    skimage_color = None

# === LOAD CONFIG ===
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

subtract_config = config['subtract']
resolution_config = config['resolution']

# === INPUT ===
parser = argparse.ArgumentParser(description='Subtract background from video to isolate creature across all frames (FAST version)')
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
parser.add_argument('--workers', type=int, default=4,
                    help='Number of parallel workers for frame processing. Default: 4')
args = parser.parse_args()

# Apply config defaults for unspecified args
threshold = args.threshold if args.threshold is not None else subtract_config['threshold']
boost = args.boost if args.boost is not None else subtract_config['boost']
feather_radius = args.feather if args.feather is not None else subtract_config['feather_radius']
diff_mode = args.mode if args.mode is not None else subtract_config['diff_mode']
output_mode = args.output if args.output is not None else subtract_config['output_mode']
min_alpha = subtract_config['min_alpha']
aspect_tolerance = resolution_config['aspect_tolerance']

# Pre-compute Gaussian kernel size for OpenCV (must be odd)
if feather_radius > 0:
    kernel_size = feather_radius * 2 + 1
else:
    kernel_size = 0


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
    try:
        frame_count = int(parts[3])
    except (ValueError, IndexError):
        frame_count = None
    return width, height, fps, frame_count


def process_frame_fast(gen_arr, bg_arr, threshold, boost, feather_radius, kernel_size, diff_mode, min_alpha):
    """Process a single frame using background subtraction (optimized version)."""
    # === COMPUTE DIFFERENCE MASK ===
    if diff_mode == 'rgb':
        diff = np.abs(gen_arr - bg_arr)
        diff_magnitude = np.max(diff, axis=2)

    elif diff_mode == 'lab':
        if skimage_color is None:
            raise ImportError("skimage is required for LAB mode. Install with: pip install scikit-image")
        gen_lab = skimage_color.rgb2lab(gen_arr / 255.0)
        bg_lab = skimage_color.rgb2lab(bg_arr / 255.0)
        diff_magnitude = np.sqrt(np.sum((gen_lab - bg_lab) ** 2, axis=2))
        diff_magnitude = np.clip(diff_magnitude * 2.55, 0, 255)

    elif diff_mode == 'luminance':
        gen_lum = 0.299 * gen_arr[:,:,0] + 0.587 * gen_arr[:,:,1] + 0.114 * gen_arr[:,:,2]
        bg_lum = 0.299 * bg_arr[:,:,0] + 0.587 * bg_arr[:,:,1] + 0.114 * bg_arr[:,:,2]
        diff_magnitude = np.abs(gen_lum - bg_lum)

    # === CREATE MASK ===
    mask = (diff_magnitude > threshold).astype(np.float32)

    # Use OpenCV GaussianBlur instead of PIL (much faster)
    if feather_radius > 0:
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    mask = np.clip(mask, min_alpha, 1.0)

    # === EXTRACT CREATURE ===
    creature_diff = gen_arr - bg_arr
    creature_diff = np.clip(creature_diff, 0, 255)
    creature_diff = creature_diff * boost
    creature = creature_diff * mask[:, :, np.newaxis]
    creature = np.clip(creature, 0, 255).astype(np.uint8)

    return creature, mask


# === LOAD BACKGROUND ===
print(f'Loading background frame (reference): {args.background_frame}')
background = Image.open(args.background_frame).convert('RGB')
bg_size = background.size
bg_aspect = get_aspect_ratio(bg_size)
print(f'  Reference size: {bg_size[0]}x{bg_size[1]} (aspect: {bg_aspect:.4f})')

# Ensure dimensions are divisible by 2 (required by most video codecs)
output_width = bg_size[0] if bg_size[0] % 2 == 0 else bg_size[0] - 1
output_height = bg_size[1] if bg_size[1] % 2 == 0 else bg_size[1] - 1
if (output_width, output_height) != bg_size:
    print(f'  Adjusting to {output_width}x{output_height} (codec requires even dimensions)')
    background = background.crop((0, 0, output_width, output_height))
    bg_size = (output_width, output_height)

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
    output_path = f"{base}_creature_{params_str}_alpha.mov"
    pix_fmt_out = 'yuva444p10le'
    output_codec = 'prores_ks'
    extra_output_args = ['-profile:v', '4444']
else:
    output_path = f"{base}_creature_{params_str}.mp4"
    pix_fmt_out = 'yuv420p'
    output_codec = args.codec
    extra_output_args = ['-crf', str(args.crf)]

preview_path = f"{base}_preview_{params_str}.mp4" if args.preview else None

# === STREAMING PIPELINE ===
print(f'\nStarting streaming pipeline...')
print(f'  Parameters: threshold={threshold}, boost={boost}, feather={feather_radius}, mode={diff_mode}')
print(f'  Workers: {args.workers}')

# Input pipe from FFmpeg (read raw frames)
input_cmd = [
    'ffmpeg', '-i', args.input_video,
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-v', 'error',
    '-'
]

# Output pipe to FFmpeg (write processed frames)
if output_mode == 'alpha':
    output_pix_fmt_in = 'rgba'
    bytes_per_pixel_out = 4
else:
    output_pix_fmt_in = 'rgb24'
    bytes_per_pixel_out = 3

output_cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{bg_size[0]}x{bg_size[1]}',
    '-pix_fmt', output_pix_fmt_in,
    '-r', str(fps),
    '-i', '-',
    '-c:v', output_codec,
    *extra_output_args,
    '-pix_fmt', pix_fmt_out,
    '-v', 'error',
    output_path
]

# Preview pipe (optional)
preview_cmd = None
if args.preview:
    preview_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{bg_size[0]}x{bg_size[1]}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', args.codec,
        '-crf', str(args.crf),
        '-pix_fmt', 'yuv420p',
        '-v', 'error',
        preview_path
    ]

# Calculate frame size
frame_size_in = width * height * 3  # RGB24 input
frame_size_out = bg_size[0] * bg_size[1] * bytes_per_pixel_out

# Start processes
input_process = subprocess.Popen(input_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output_process = subprocess.Popen(output_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
preview_process = subprocess.Popen(preview_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) if preview_cmd else None

alignment_logged = False
frame_num = 0
total_frames = frame_count if frame_count else "unknown"

print(f'\nProcessing frames (streaming)...')

try:
    while True:
        # Read raw frame from input
        raw_frame = input_process.stdout.read(frame_size_in)
        if len(raw_frame) < frame_size_in:
            break  # End of video

        frame_num += 1
        if frame_num % 30 == 0 or frame_num == 1:
            print(f'  Processing frame {frame_num}/{total_frames}')

        # Convert to numpy array
        frame_arr = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

        # Align if needed
        if needs_alignment:
            frame_pil = Image.fromarray(frame_arr)
            frame_pil, align_info = align_to_reference(frame_pil, background, aspect_tolerance)
            if not alignment_logged:
                print_alignment_info(align_info, prefix='    ')
                alignment_logged = True
            frame_arr = np.array(frame_pil)

        gen_arr = frame_arr.astype(np.float32)

        # Process frame
        creature, mask = process_frame_fast(
            gen_arr, bg_arr, threshold, boost, feather_radius, kernel_size, diff_mode, min_alpha
        )

        # Write to output
        if output_mode == 'alpha':
            alpha_channel = (mask * 255).astype(np.uint8)
            rgba = np.dstack([creature, alpha_channel])
            data_to_write = rgba.tobytes()
        else:
            data_to_write = creature.tobytes()

        # Check if output process is still alive
        if output_process.poll() is not None:
            stderr_output = output_process.stderr.read().decode() if output_process.stderr else ""
            raise RuntimeError(f"FFmpeg output process died unexpectedly.\nFFmpeg stderr: {stderr_output}")

        output_process.stdin.write(data_to_write)

        # Write preview if requested
        if preview_process:
            composite = np.clip(bg_arr + creature.astype(np.float32), 0, 255).astype(np.uint8)
            preview_process.stdin.write(composite.tobytes())

    print(f'  Processed {frame_num} frames total')

except BrokenPipeError as e:
    # Get FFmpeg error output
    stderr_output = output_process.stderr.read().decode() if output_process.stderr else ""
    print(f"\n*** FFmpeg Error ***")
    print(f"BrokenPipeError at frame {frame_num}")
    if stderr_output:
        print(f"FFmpeg stderr: {stderr_output}")
    raise

finally:
    # Close pipes and wait for processes
    if input_process.stdout:
        input_process.stdout.close()
    if output_process.stdin:
        output_process.stdin.close()
    if preview_process and preview_process.stdin:
        preview_process.stdin.close()

    input_process.wait()
    ret = output_process.wait()
    if ret != 0:
        stderr_output = output_process.stderr.read().decode() if output_process.stderr else ""
        print(f"\n*** FFmpeg output process exited with code {ret} ***")
        if stderr_output:
            print(f"FFmpeg stderr: {stderr_output}")
    if preview_process:
        preview_process.wait()

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
