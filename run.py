#!/usr/bin/env python3
"""
WZRD Config Runner

Run wzrd tools from JSON configuration files.

Usage:
    python run.py config.json
    python run.py config_img_darken.json
    python run.py config_video_background_remove.json
"""

import argparse
import json
import sys
from pathlib import Path

from wzrd import (
    darken_image_file,
    subtract_background_file,
    subtract_background_video,
    crop_islands_file,
    reproject_video,
    reproject_videos_batch,
)


def run_darken(config: dict) -> dict:
    """Run image darkening tool."""
    input_path = Path(config["input"])
    output_path = config.get("output")
    settings = config.get("settings", {})

    # Auto-generate output path if not provided (same logic as CLI)
    if output_path is None:
        gamma = settings.get("gamma", 1.5)
        output_path = str(input_path.with_name(f"{input_path.stem}_darkened_{gamma:.2f}.jpg"))

    img, info = darken_image_file(
        input_path,
        output_path=output_path,
        gamma=settings.get("gamma"),
        max_brightness=settings.get("max_brightness"),
        target_aspect=settings.get("target_aspect"),
        base_resolution=settings.get("base_resolution"),
        normalize=settings.get("normalize", True),
    )

    return info


def run_subtract_image(config: dict) -> dict:
    """Run image background subtraction tool."""
    input_path = Path(config["input"])
    background_path = config["background"]
    output_path = config.get("output")
    settings = config.get("settings", {})

    # Auto-generate output path if not provided (same logic as CLI)
    if output_path is None:
        threshold = settings.get("threshold", 10)
        boost = settings.get("boost", 1.1)
        feather = settings.get("feather_radius", 4)
        mode = settings.get("diff_mode", "luminance")
        output_mode = settings.get("output_mode", "additive")
        params_str = f"t{threshold}_b{boost:.1f}_f{feather}_{mode}"
        suffix = "_alpha.png" if output_mode == "alpha" else ".png"
        output_path = str(input_path.with_name(f"{input_path.stem}_creature_{params_str}{suffix}"))

    img, info = subtract_background_file(
        input_path,
        background_path,
        output_path=output_path,
        threshold=settings.get("threshold"),
        boost=settings.get("boost"),
        feather_radius=settings.get("feather_radius"),
        diff_mode=settings.get("diff_mode"),
        output_mode=settings.get("output_mode"),
        preview=settings.get("preview", False),
    )

    return info


def run_subtract_video(config: dict) -> dict:
    """Run video background subtraction tool."""
    input_path = config["input"]
    background_path = config["background"]
    output_path = config.get("output")
    settings = config.get("settings", {})

    def progress_callback(frame: int, total: int):
        if total > 0:
            pct = frame / total * 100
            print(f"\rProcessing: {frame}/{total} ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\rProcessing frame {frame}...", end="", flush=True)

    info = subtract_background_video(
        input_path,
        background_path,
        output_path=output_path,
        threshold=settings.get("threshold"),
        boost=settings.get("boost"),
        feather_radius=settings.get("feather_radius"),
        diff_mode=settings.get("diff_mode"),
        output_mode=settings.get("output_mode"),
        preview=settings.get("preview", False),
        crf=settings.get("crf", 18),
        codec=settings.get("codec", "libx264"),
        progress_callback=progress_callback,
    )

    print()  # newline after progress
    return info


def run_crop_islands(config: dict) -> dict:
    """Run island extraction tool."""
    input_path = config["input"]
    output_dir = config.get("output")
    settings = config.get("settings", {})

    islands, output_path = crop_islands_file(
        input_path,
        output_dir=output_dir,
        min_area=settings.get("min_area", 0),
        connectivity=settings.get("connectivity", 8),
    )

    return {
        "islands_found": len(islands),
        "output_dir": str(output_path),
        "islands": islands,
    }


def run_reproject(config: dict) -> dict:
    """Run video reprojection tool."""
    settings = config.get("settings", {})

    def progress_callback(frame: int, total: int):
        if total > 0:
            pct = frame / total * 100
            print(f"\rProcessing: {frame}/{total} ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\rProcessing frame {frame}...", end="", flush=True)

    def batch_progress_callback(video_num: int, total_videos: int, frame: int, total: int):
        if total > 0:
            pct = frame / total * 100
            print(f"\r[{video_num}/{total_videos}] Processing: {frame}/{total} ({pct:.1f}%)", end="", flush=True)
        else:
            print(f"\r[{video_num}/{total_videos}] Processing frame {frame}...", end="", flush=True)

    # Check if we have a list of videos (batch mode) or single video
    if "videos" in config:
        # Batch mode: list of {video, metadata, output} dicts
        video_json_pairs = []
        for item in config["videos"]:
            video_path = item["video"]
            metadata_path = item["metadata"]
            video_json_pairs.append((video_path, metadata_path))

        results = reproject_videos_batch(
            video_json_pairs=video_json_pairs,
            output_dir=config.get("output_dir"),
            canvas_width=settings.get("canvas_width", 1920),
            canvas_height=settings.get("canvas_height", 1080),
            crf=settings.get("crf", 18),
            codec=settings.get("codec", "libx264"),
            progress_callback=batch_progress_callback,
        )

        print()  # newline after progress
        return {
            "videos_processed": len(results),
            "outputs": [r["output_video"] for r in results],
            "results": results,
        }

    else:
        # Single video mode
        input_path = config["input"]
        metadata_path = config["metadata"]
        output_path = config.get("output")

        info = reproject_video(
            video_path=input_path,
            island_metadata=metadata_path,
            output_path=output_path,
            canvas_width=settings.get("canvas_width", 1920),
            canvas_height=settings.get("canvas_height", 1080),
            crf=settings.get("crf", 18),
            codec=settings.get("codec", "libx264"),
            progress_callback=progress_callback,
        )

        print()  # newline after progress
        return info


# Tool registry mapping tool names to functions
TOOLS = {
    "darken": run_darken,
    "darken_image": run_darken,
    "subtract": run_subtract_image,
    "subtract_image": run_subtract_image,
    "subtract_background": run_subtract_image,
    "subtract_video": run_subtract_video,
    "subtract_background_video": run_subtract_video,
    "video_subtract": run_subtract_video,
    "crop_islands": run_crop_islands,
    "islands": run_crop_islands,
    "reproject": run_reproject,
    "reproject_video": run_reproject,
    "reproject_videos": run_reproject,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run WZRD tools from JSON configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example config files:

  config_img_darken.json:
  {
    "tool": "darken",
    "input": "surface.jpg",
    "output": "surface_dark.jpg",
    "settings": {
      "gamma": 1.5,
      "max_brightness": 0.15,
      "target_aspect": "16:9"
    }
  }

  config_video_background_remove.json:
  {
    "tool": "subtract_video",
    "input": "animation.mp4",
    "background": "background.png",
    "output": "creature.mp4",
    "settings": {
      "threshold": 10,
      "boost": 1.1,
      "preview": true
    }
  }

  config_reproject.json:
  {
    "tool": "reproject",
    "videos": [
      {"video": "island_1.mp4", "metadata": "island_1.json"},
      {"video": "island_2.mp4", "metadata": "island_2.json"}
    ],
    "output_dir": "reprojected/",
    "settings": {
      "canvas_width": 1920,
      "canvas_height": 1080
    }
  }

Available tools:
  - darken, darken_image
  - subtract, subtract_image, subtract_background
  - subtract_video, subtract_background_video, video_subtract
  - crop_islands, islands
  - reproject, reproject_video, reproject_videos
""",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    # Validate required fields
    if "tool" not in config:
        print("Error: Config must specify 'tool' field", file=sys.stderr)
        sys.exit(1)

    tool_name = config["tool"].lower()
    if tool_name not in TOOLS:
        print(f"Error: Unknown tool '{tool_name}'", file=sys.stderr)
        print(f"Available tools: {', '.join(sorted(set(TOOLS.keys())))}", file=sys.stderr)
        sys.exit(1)

    if "input" not in config:
        print("Error: Config must specify 'input' field", file=sys.stderr)
        sys.exit(1)

    # Run tool
    print(f"Running {tool_name}...")
    if args.verbose:
        print(f"Config: {json.dumps(config, indent=2)}")

    try:
        tool_func = TOOLS[tool_name]
        result = tool_func(config)

        print("Done!")
        if args.verbose:
            print(f"Result: {json.dumps(result, indent=2, default=str)}")
        elif "output_path" in result:
            print(f"Output: {result['output_path']}")
        elif "output_video" in result:
            print(f"Output: {result['output_video']}")
        elif "output_dir" in result:
            print(f"Output: {result['output_dir']}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
