import torch
import numpy as np
from PIL import Image
import cv2
import os
import random
from dataclasses import dataclass
from typing import Tuple, List, Union
from pathlib import Path
from enum import Enum

class StartPosition(Enum):
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    TOP_CENTER = "top_center"
    BOTTOM_CENTER = "bottom_center"
    LEFT_CENTER = "left_center"
    RIGHT_CENTER = "right_center"
    RANDOM = "random"

@dataclass
class FillConfig:
    target_size: int = 264
    threshold_value: int = 128
    seed_radius: int = 5
    noise_range: Tuple[float, float] = (0.0, 1.2)
    growth_threshold: float = 0.6
    stability_threshold: int = 20
    saturation_window: int = 15
    saturation_threshold: float = 0.001
    fps: float = 20.0
    starting_position: StartPosition = StartPosition.RANDOM
    position_randomness: float = 1.0
    island_connection_radius: float = 0.005
    active_region_padding: int = 3
    invert_input: bool = False
    num_seeds: int = 1
    color_channel: str = "luminance"
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    loop: bool = True
    hold_final_frame_fraction: float = 0.25

class OrganicFill:
    def __init__(self, config: FillConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.fill_history = []
        self.frame_count = 0
        self.seed_location = None
        
    def _process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Connect isolated regions in the mask"""

        if self.config.island_connection_radius == 0:
            return mask.to(device=self.device, dtype=torch.float32)
        
        mask_np = mask.cpu().numpy().astype(np.uint8)
        kernel_size = max(3, int(self.config.island_connection_radius * max(mask_np.shape)))
        kernel_size += (kernel_size % 2 == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return torch.from_numpy(
            np.maximum(
                mask_np,
                cv2.erode(
                    cv2.dilate(mask_np, kernel, iterations=2),
                    kernel, iterations=1
                )
            )
        ).to(device=self.device, dtype=torch.float32)

    def load_image(self, image_path: Path) -> None:
        """Initialize the system from an input image"""

        def _calculate_resize_dims(original_size: Tuple[int, int], max_dim: int) -> Tuple[int, int]:
            """Calculate new dimensions maintaining aspect ratio with max dimension constraint"""
            width, height = original_size
            if width >= height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            return (new_width, new_height)

        def _extract_channel(img_array: np.ndarray, channel: str) -> np.ndarray:
            """Extract specified channel from RGB image"""
            if len(img_array.shape) == 2:  # Already grayscale
                return img_array

            if channel == "red":
                return img_array[:, :, 0]
            elif channel == "green":
                return img_array[:, :, 1]
            elif channel == "blue":
                return img_array[:, :, 2]
            elif channel == "random":
                import random
                return img_array[:, :, random.randint(0, 2)]
            else:  # "luminance"
                # Convert to grayscale using standard luminance formula
                return np.dot(img_array[...,:3], [0.299, 0.587, 0.114])

        # Load image and handle color channel selection
        img = Image.open(image_path)
        if self.config.color_channel == "luminance" or img.mode == 'L':
            img = img.convert('L')
            resize_dims = _calculate_resize_dims(img.size, self.config.target_size)
            img = np.array(img.resize(resize_dims))
        else:
            img = img.convert('RGB')
            resize_dims = _calculate_resize_dims(img.size, self.config.target_size)
            img_array = np.array(img.resize(resize_dims))
            img = _extract_channel(img_array, self.config.color_channel)

        # Apply inversion if requested
        if self.config.invert_input:
            img = 255 - img

        self.mask = self._process_mask(torch.tensor(img < self.config.threshold_value))
        self.height, self.width = self.mask.shape
        
        # Initialize state tensors
        self.grid = torch.zeros_like(self.mask)
        self.activity_counter = torch.zeros_like(self.mask, dtype=torch.int32)
        self.active_mask = torch.ones_like(self.mask, dtype=torch.bool)
        
        # Set initial seed
        self._place_seed()

    def _place_seed(self) -> None:
        """Place initial seed(s) in specified position"""
        h, w = self.mask.shape
        valid_coords = torch.where(self.mask > 0)

        if self.config.num_seeds == 1:
            # Single seed placement (existing logic)
            # Define region filters for different starting positions
            position_filters = {
                StartPosition.TOP_LEFT: lambda y, x: (y < h//2) & (x < w//2),
                StartPosition.TOP_RIGHT: lambda y, x: (y < h//2) & (x >= w//2),
                StartPosition.BOTTOM_LEFT: lambda y, x: (y >= h//2) & (x < w//2),
                StartPosition.BOTTOM_RIGHT: lambda y, x: (y >= h//2) & (x >= w//2),
                StartPosition.CENTER: lambda y, x: (
                    (y >= h//3) & (y < 2*h//3) & (x >= w//3) & (x < 2*w//3)
                ),
                StartPosition.TOP_CENTER: lambda y, x: (y < h//3) & (x >= w//3) & (x < 2*w//3),
                StartPosition.BOTTOM_CENTER: lambda y, x: (y >= 2*h//3) & (x >= w//3) & (x < 2*w//3),
                StartPosition.LEFT_CENTER: lambda y, x: (x < w//3) & (y >= h//3) & (y < 2*h//3),
                StartPosition.RIGHT_CENTER: lambda y, x: (x >= 2*w//3) & (y >= h//3) & (y < 2*h//3),
                StartPosition.RANDOM: lambda y, x: torch.ones_like(y, dtype=torch.bool)
            }

            # Apply position filter
            position_filter = position_filters[self.config.starting_position]
            valid_mask = position_filter(valid_coords[0], valid_coords[1])

            if not torch.any(valid_mask):
                # Fallback to center if preferred position is invalid
                valid_mask = position_filters[StartPosition.CENTER](valid_coords[0], valid_coords[1])
                if not torch.any(valid_mask):
                    valid_mask = position_filters[StartPosition.RANDOM](valid_coords[0], valid_coords[1])

            # Get valid points in the sampling region
            valid_points = torch.stack([coord[valid_mask] for coord in valid_coords])

            if self.config.position_randomness == 0.0:
                # Use exact center of the sampling region
                center_y = valid_points[0].float().mean()
                center_x = valid_points[1].float().mean()
                seed_y, seed_x = center_y, center_x
            elif self.config.position_randomness == 1.0:
                # Use fully random position in the sampling region
                seed_idx = torch.randint(valid_points.shape[1], (1,))
                seed_y, seed_x = valid_points[:, seed_idx]
            else:
                # Interpolate between center and random position
                center_y = valid_points[0].float().mean()
                center_x = valid_points[1].float().mean()

                seed_idx = torch.randint(valid_points.shape[1], (1,))
                random_y, random_x = valid_points[:, seed_idx].float()

                # Linear interpolation: center + randomness * (random - center)
                seed_y = center_y + self.config.position_randomness * (random_y - center_y)
                seed_x = center_x + self.config.position_randomness * (random_x - center_x)

            seed_positions = [(seed_x, seed_y)]

        else:
            # Multiple seed placement
            # Get all valid points
            valid_points = torch.stack([coord for coord in valid_coords])

            if self.config.position_randomness == 0.0:
                # Symmetrical placement around image center
                center_y = h / 2.0
                center_x = w / 2.0

                seed_positions = []
                for i in range(self.config.num_seeds):
                    # Create symmetrical pattern
                    angle = 2 * np.pi * i / self.config.num_seeds
                    # Use a radius that's 1/4 of the image diagonal
                    radius = min(h, w) / 4.0

                    seed_x = center_x + radius * np.cos(angle)
                    seed_y = center_y + radius * np.sin(angle)

                    # Clamp to valid mask area if possible
                    seed_x = max(0, min(w-1, seed_x))
                    seed_y = max(0, min(h-1, seed_y))

                    seed_positions.append((seed_x, seed_y))

            elif self.config.position_randomness == 1.0:
                # Fully random positions
                seed_positions = []
                for _ in range(self.config.num_seeds):
                    seed_idx = torch.randint(valid_points.shape[1], (1,))
                    seed_y, seed_x = valid_points[:, seed_idx]
                    seed_positions.append((float(seed_x), float(seed_y)))

            else:
                # Interpolate between symmetric and random
                center_y = h / 2.0
                center_x = w / 2.0

                seed_positions = []
                for i in range(self.config.num_seeds):
                    # Symmetric position
                    angle = 2 * np.pi * i / self.config.num_seeds
                    radius = min(h, w) / 4.0
                    sym_x = center_x + radius * np.cos(angle)
                    sym_y = center_y + radius * np.sin(angle)

                    # Random position
                    seed_idx = torch.randint(valid_points.shape[1], (1,))
                    rand_y, rand_x = valid_points[:, seed_idx].float()

                    # Interpolate
                    seed_x = sym_x + self.config.position_randomness * (rand_x - sym_x)
                    seed_y = sym_y + self.config.position_randomness * (rand_y - sym_y)

                    seed_positions.append((float(seed_x), float(seed_y)))

        # Create seeds at all positions
        y, x = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )

        for seed_x, seed_y in seed_positions:
            seed = ((y - seed_y)**2 + (x - seed_x)**2 <= self.config.seed_radius**2)
            self.grid[seed & (self.mask > 0)] = 1.0

        # Store first seed location as percentage of width/height (for compatibility)
        if seed_positions:
            first_seed_x, first_seed_y = seed_positions[0]
            self.seed_location = (first_seed_x / self.width, first_seed_y / self.height)

    def _update_activity(self, new_growth: torch.Tensor) -> None:
        """Update activity tracking and active regions"""
        # Reset counter for new growth areas
        self.activity_counter[new_growth] = 0
        
        # Increment counter for unchanged areas
        self.activity_counter[~new_growth] += 1
        
        # Create mask of recently active regions with padding
        recent_activity = self.activity_counter < self.config.stability_threshold
        recent_np = recent_activity.cpu().numpy()
        kernel = np.ones((2*self.config.active_region_padding + 1,)*2, np.uint8)
        padded_active = cv2.dilate(recent_np.astype(np.uint8), kernel)
        
        # Update active mask
        self.active_mask = (
            torch.from_numpy(padded_active).to(self.device).bool() & 
            (self.grid < 1) & 
            (self.mask > 0)
        )

    def step(self) -> None:
        """Perform one growth step"""
        self.frame_count += 1
        
        # Calculate growth boundary
        padded = torch.nn.functional.pad(self.grid, (1,1,1,1))
        neighbors = torch.nn.functional.unfold(
            padded.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            padding=0
        ).squeeze().view(9, self.height, self.width)
        
        boundary = (
            (self.grid == 0) & 
            (self.mask > 0) & 
            ((neighbors.sum(0) - neighbors[4]) > 0) &
            self.active_mask
        )
        
        # Apply growth with noise
        noise = torch.rand_like(self.grid) * (
            self.config.noise_range[1] - self.config.noise_range[0]
        ) + self.config.noise_range[0]
        
        new_growth = boundary & (noise > self.config.growth_threshold)
        self.grid[new_growth] = 1
        
        # Update activity tracking
        self._update_activity(new_growth)
        
        # Update fill history
        self.fill_history.append(self.compute_fill_ratio())

    def compute_fill_ratio(self) -> float:
        """Calculate current fill ratio"""
        return (self.grid * self.mask).sum() / self.mask.sum()

    def is_complete(self) -> bool:
        """Check if growth is complete"""
        if len(self.fill_history) < self.config.saturation_window:
            return False
            
        recent_change = max(
            abs(self.fill_history[i] - self.fill_history[i-1])
            for i in range(-self.config.saturation_window + 1, 0)
        )
        
        return (recent_change < self.config.saturation_threshold and 
                not torch.any((self.grid == 0) & self.active_mask & (self.mask > 0)))

    def get_frame(self) -> np.ndarray:
        """Get current frame for video"""
        return (self.grid.cpu().numpy() * 255).astype(np.uint8)

def resample_frames(frames: List[np.ndarray], target_n_frames: int) -> List[np.ndarray]:
    """Resample frames to target count using temporal interpolation with blending

    Args:
        frames: List of input frames (numpy arrays)
        target_n_frames: Desired number of output frames

    Returns:
        List of resampled frames with smooth temporal interpolation
    """
    if not frames:
        return []

    if len(frames) == target_n_frames:
        return frames.copy()

    if target_n_frames == 1:
        return [frames[len(frames) // 2]]  # Return middle frame

    resampled_frames = []
    n_input = len(frames)

    for i in range(target_n_frames):
        # Map target frame index to continuous source timeline
        # Use (n_input - 1) to map [0, target_n_frames-1] -> [0, n_input-1]
        source_pos = i * (n_input - 1) / (target_n_frames - 1)

        # Get integer part (frame index) and fractional part (blend weight)
        frame_idx = int(source_pos)
        blend_weight = source_pos - frame_idx

        # Handle edge case for last frame
        if frame_idx >= n_input - 1:
            resampled_frames.append(frames[-1].copy())
            continue

        # Blend between current and next frame
        if blend_weight < 1e-6:  # Essentially no blending needed
            resampled_frames.append(frames[frame_idx].copy())
        else:
            # Linear interpolation between frames
            frame1 = frames[frame_idx].astype(np.float32)
            frame2 = frames[frame_idx + 1].astype(np.float32)

            blended = frame1 * (1 - blend_weight) + frame2 * blend_weight
            resampled_frames.append(blended.astype(np.uint8))

    return resampled_frames

def grow_starting_frames(starting_frame: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Generate gradually evolving frames that build up to the starting frame

    Args:
        starting_frame: The target starting frame to evolve towards
        num_frames: Number of frames to generate (e.g., 10)

    Returns:
        List of frames that gradually evolve into the starting frame
    """
    if num_frames <= 0:
        return []

    frames = []
    height, width = starting_frame.shape

    # Create a completely black starting frame
    base_frame = np.zeros_like(starting_frame)

    # Find all white pixels in the starting frame
    white_pixels = np.where(starting_frame > 0)
    total_white_pixels = len(white_pixels[0])

    if total_white_pixels == 0:
        # If no white pixels, just return empty frames
        return [base_frame.copy() for _ in range(num_frames)]

    # For each frame, gradually add more white pixels
    for frame_idx in range(num_frames):
        current_frame = base_frame.copy()

        # Calculate how many pixels should be white in this frame
        progress = (frame_idx + 1) / num_frames
        # Use a smooth curve (ease-in) for more natural growth
        smooth_progress = progress * progress
        pixels_to_show = int(smooth_progress * total_white_pixels)

        if pixels_to_show > 0:
            # Randomly select which pixels to make white
            indices = np.random.choice(total_white_pixels, pixels_to_show, replace=False)
            selected_y = white_pixels[0][indices]
            selected_x = white_pixels[1][indices]
            current_frame[selected_y, selected_x] = 255

        frames.append(current_frame)

    return frames

def generate_reverse_frames(forward_frames: List[np.ndarray]) -> List[np.ndarray]:
    """Generate reverse animation frames following the same directional pattern as forward

    Args:
        forward_frames: List of frames from the forward animation

    Returns:
        List of frames for the reverse animation (white to black in same direction)
    """
    if not forward_frames:
        return []

    reverse_frames = []

    # Start with the last frame (fully filled)
    current_frame = forward_frames[-1].copy()
    reverse_frames.append(current_frame.copy())

    # Apply the same growth pattern as forward, but turn pixels black instead
    for i in range(1, len(forward_frames)):
        current_forward = forward_frames[i]
        previous_forward = forward_frames[i-1]

        # Find pixels that were added in this step (became white)
        newly_white_pixels = (current_forward > previous_forward)

        # Turn those same pixels black in our reverse frame (same direction as forward)
        current_frame[newly_white_pixels] = 0

        # Add this reverse frame
        reverse_frames.append(current_frame.copy())

    return reverse_frames

def create_animation(image_path: Union[str, Path], target_n_seconds: float, output_path: str = None, config: FillConfig = None, save_final_frame: bool = False) -> None:
    """Create organic fill animation from image with specified duration

    Args:
        image_path: Path to input image
        target_n_seconds: Target duration in seconds (required)
        output_path: Output video path (auto-generated if None)
        config: Animation configuration
        save_final_frame: Whether to save final frame as image
    """
    image_path = Path(image_path)
    config = config or FillConfig()

    fill = OrganicFill(config)
    fill.load_image(image_path)

    # If no output path provided, create one based on input path and parameters
    if output_path is None:
        seed_x_pct, seed_y_pct = fill.seed_location
        param_str = f"gt{config.growth_threshold:.1f}_pr{config.position_randomness:.1f}_sr{config.seed_radius}"
        location_str = f"loc{seed_x_pct:.2f}x{seed_y_pct:.2f}"
        loop_suffix = "_loop" if config.loop else ""
        output_path = str(image_path.with_suffix('')) + f"_{param_str}_{location_str}{loop_suffix}_vid.mp4"

    video_size = (fill.width, fill.height)

    # Always collect frames first (needed for resampling and/or looping)
    forward_frames = []

    # Generate forward animation (fill phase)
    print("Generating forward animation...")
    while not fill.is_complete():
        fill_ratio = fill.compute_fill_ratio()
        fill.step()
        frame = fill.get_frame()
        forward_frames.append(frame.copy())

        if fill.frame_count % 100 == 0:
            active_count = fill.active_mask.sum().item()
            print(f"Frame {fill.frame_count}: Fill {fill_ratio:.2%}, Active regions: {active_count}")

    # Generate gradual starting frames to avoid abrupt beginning
    if forward_frames:
        print("Generating gradual starting frames...")
        starting_frames = grow_starting_frames(forward_frames[0], 10)
        # Combine starting frames with the main animation for the complete forward sequence
        complete_forward_frames = starting_frames + forward_frames.copy()
        all_frames = complete_forward_frames.copy()
    else:
        complete_forward_frames = forward_frames.copy()
        all_frames = forward_frames.copy()

    # Generate reverse animation if looping is enabled
    if config.loop:
        # Hold the final frame if requested
        if config.hold_final_frame_fraction > 0:
            hold_frames = int(config.hold_final_frame_fraction * len(complete_forward_frames))
            if hold_frames > 0:
                print(f"Adding {hold_frames} hold frames...")
                final_frame = complete_forward_frames[-1].copy()
                for _ in range(hold_frames):
                    all_frames.append(final_frame)

        print("Generating reverse animation for seamless loop...")
        reverse_frames = generate_reverse_frames(complete_forward_frames)
        all_frames.extend(reverse_frames)

        # Hold the final reversed frame (= first frame) for the same duration
        if config.hold_final_frame_fraction > 0:
            hold_frames = int(config.hold_final_frame_fraction * len(forward_frames)) // 2
            if hold_frames > 0:
                print(f"Adding {hold_frames} hold frames at end...")
                final_reversed_frame = reverse_frames[-1].copy()
                for _ in range(hold_frames):
                    all_frames.append(final_reversed_frame)

    # Calculate target frame count from target duration and fps
    target_n_frames = int(target_n_seconds * config.fps)
    original_frame_count = len(all_frames)

    print(f"Resampling {original_frame_count} frames to {target_n_frames} frames ({target_n_seconds}s @ {config.fps} fps)...")
    all_frames = resample_frames(all_frames, target_n_frames)

    # Write all frames to video
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        config.fps,
        video_size,
        False
    )

    try:
        for i, frame in enumerate(all_frames):
            out.write(frame)
            if (i + 1) % 100 == 0:
                progress = (i + 1) / len(all_frames)
                print(f"Writing frame {i + 1}/{len(all_frames)}: {progress:.1%} complete")

    finally:
        out.release()

    if save_final_frame:
        # Save final frame as an image
        final_image_path = str(Path(output_path).with_suffix('')) + "_final.png"
        final_frame = fill.get_frame()
        cv2.imwrite(final_image_path, final_frame)
        print(f"Final frame saved to {final_image_path}")

    # Convert to web-compatible format
    temp_path = str(Path(output_path).with_suffix('.temp.mp4'))
    os.rename(output_path, temp_path)
    os.system(f'ffmpeg -i {temp_path} -c:v h264_videotoolbox {output_path} -y')
    os.remove(temp_path)

    total_frames = len(all_frames)
    loop_info = " (loopable)" if config.loop else ""
    resampling_info = f" (resampled from {original_frame_count})"
    print(f"Animation completed: {total_frames} frames{loop_info}{resampling_info}, saved to {output_path}")

def process_images(input_path: Union[str, Path], target_n_seconds: float, config: FillConfig = None) -> None:
    """Process a single image or all images in a directory

    Args:
        input_path: Path to image file or directory
        target_n_seconds: Target duration in seconds for each animation
        config: Animation configuration
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        print(f"Processing single image: {input_path}")
        create_animation(input_path, target_n_seconds, config=config)
    elif input_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in input_path.iterdir()
                       if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {input_path}")
            return

        print(f"Processing {len(image_files)} images from directory: {input_path}")
        for img_file in image_files:
            print(f"Processing: {img_file}")
            create_animation(img_file, target_n_seconds, config=config)

"""
Organic Fill Algorithm - Creates animated organic growth patterns from images

Main Parameters:
- target_size: Maximum dimension for resized image (default: 264)
- threshold_value: Grayscale threshold for creating mask (default: 128)
- seed_radius: Size of initial growth seed (default: 5)
- growth_threshold: Controls growth probability (0.0-1.0, default: 0.6)
- noise_range: Random variation in growth (default: 0.0-1.2)
- starting_position: Where growth begins (center, corners, random, etc.)
- island_connection_radius: Connects isolated regions (0.005 = moderate connection)
- active_region_padding: Expansion of active growth areas (default: 3)
- saturation_threshold: Growth completion sensitivity (default: 0.001)
- fps: Output video frame rate (default: 20.0)
"""

if __name__ == "__main__":
    config = FillConfig(
        target_size=512,
        growth_threshold=0.5,
        noise_range=(0.0, 1.2),
        island_connection_radius=0.0005,
        starting_position=StartPosition.CENTER,
        position_randomness=0.0,
        saturation_threshold=0.0001,
        active_region_padding=3,
        fps=24,
        loop=True  # Enable looping for VJ use
    )
    
    import sys
    if len(sys.argv) > 2:
        process_images(sys.argv[1], float(sys.argv[2]), config)
    else:
        print("Usage: python fill_polygon.py <image_path_or_directory> <target_seconds>")
        print("Example: python fill_polygon.py tree2.jpg 5.0")
        # Example with hardcoded path
        process_images('tree2.jpg', 5.0, config)