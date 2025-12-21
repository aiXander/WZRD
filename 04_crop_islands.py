import cv2
import numpy as np
import json
import os
import argparse

# === INPUT ===
parser = argparse.ArgumentParser(description='Extract individual island crops from a binary segmentation map')
parser.add_argument('input_image', help='Path to the binary segmentation map image (white regions on black background)')
parser.add_argument('--output', '-o', type=str, default='islands_output',
                    help='Output directory for cropped islands and JSON map. Default: islands_output')
parser.add_argument('--min-area', type=int, default=0,
                    help='Minimum area in pixels to include an island (filters noise). Default: 0')
parser.add_argument('--connectivity', type=int, choices=[4, 8], default=8,
                    help='Pixel connectivity for finding connected components (4=orthogonal, 8=diagonal). Default: 8')
args = parser.parse_args()


def process_segmentation_map(image_path, output_dir, min_area=0, connectivity=8):
    # 1. Setup Output Directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Load the Image
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Ensure binary (Make sure white is 255 and black is 0)
    # Thresholding cleans up any compression artifacts
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. Find Connected Components (Islands)
    # connectivity=8 checks diagonal pixels, connectivity=4 checks only orthogonal.
    # 8 is usually better for 'islands'.
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=connectivity)

    # num_labels includes the background (label 0), so we subtract 1 for the actual object count
    island_count = num_labels - 1
    print(f"Analysis complete. Found {island_count} unique white regions.")

    islands_data = []

    # 4. Iterate through labels (skipping 0, which is the black background)
    for i in range(1, num_labels):
        # Extract stats for this label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter out tiny noise if min_area is set
        if area < min_area:
            continue

        # 5. Crop the rectangle
        # We slice the binary image [y:y+h, x:x+w]
        crop = binary_img[y:y+h, x:x+w]

        # Construct filename
        filename = f"island_{i}.png"
        save_path = os.path.join(output_dir, filename)

        # Save to disk
        cv2.imwrite(save_path, crop)
        
        # 6. Store Coordinates
        island_info = {
            "id": i,
            "filename": filename,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area_pixels": int(area)
        }
        islands_data.append(island_info)
        print(f"Processed Island {i}: Saved to {filename} | Pos: ({x},{y}) Size: {w}x{h}")

    # 7. Save JSON map
    json_path = os.path.join(output_dir, "islands.json")
    with open(json_path, 'w') as f:
        json.dump(islands_data, f, indent=4)

    print(f"\nSuccess! JSON map saved to: {json_path}")
    print(f"Individual crops saved in: {output_dir}/")

# --- Main ---
if __name__ == "__main__":
    if not os.path.exists(args.input_image):
        print(f"Error: Input file '{args.input_image}' not found.")
        exit(1)

    # Place output directory in the same folder as the source image
    input_dir = os.path.dirname(os.path.abspath(args.input_image))
    output_dir = os.path.join(input_dir, args.output)

    process_segmentation_map(
        args.input_image,
        output_dir,
        min_area=args.min_area,
        connectivity=args.connectivity
    )