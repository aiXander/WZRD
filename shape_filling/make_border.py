from PIL import Image
import numpy as np

def create_bordered_image(width, height, border_fraction):
    # Calculate border width
    border = int(border_fraction * max(width, height))

    output_path=f"bordered_{border_fraction:.2f}_{max(width, height)}.png"

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate distance from each edge
    dist_from_top = y
    dist_from_bottom = height - 1 - y
    dist_from_left = x
    dist_from_right = width - 1 - x

    # Get minimum distance to any edge
    dist_from_edge = np.minimum(np.minimum(dist_from_top, dist_from_bottom),
                                np.minimum(dist_from_left, dist_from_right))

    # Create gradient based on distance from edge
    # Values: 0 (black) at edge, 255 (white) at border distance or more
    img_array = 255 - np.clip((border - dist_from_edge) * 255 / border, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    img = Image.fromarray(img_array, mode='L')

    # Save result
    img.save(output_path)
    print(f"Saved bordered image to {output_path}")


if __name__ == "__main__":
    # Example usage
    create_bordered_image(1200, 800, 0.03)
