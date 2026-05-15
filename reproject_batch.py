"""
Batch reproject rendered mp4 files onto the original 16:9 canvas.

Hardcoded region metadata for bottom region (region 0).
Change INPUT_DIR and REGION to process different folders/regions.

cd /Users/xandersteenbrugge/Documents/GitHub/Gigabrain/VJ/WZRD
python reproject_batch.py

"""

from pathlib import Path
from wzrd.reproject import reproject_video

REGION_TOP = {
    "x": 84,
    "y": -129,
    "width": 1696,
    "height": 730,
}
REGION_BOTTOM = {
    "x": 0,
    "y": 292,
    "width": 1920,
    "height": 828,
}

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR = Path("/Users/xandersteenbrugge/Visuals/pirate_ship/bottom_todo")

REGION = REGION_BOTTOM

# ── Processing ───────────────────────────────────────────────────────────────

def main():
    mp4_files = sorted(INPUT_DIR.glob("*.mp4"))
    mp4_files = [f for f in mp4_files if not f.stem.endswith("_reprojected")]

    if not mp4_files:
        print(f"No .mp4 files found in {INPUT_DIR}")
        return

    print(f"Found {len(mp4_files)} video(s) in {INPUT_DIR}")
    print(f"Region: x={REGION['x']}, y={REGION['y']}, {REGION['width']}x{REGION['height']}")
    print(f"Canvas: 1920x1080\n")

    for i, video_path in enumerate(mp4_files, 1):
        output_path = video_path.with_name(f"{video_path.stem}_reprojected.mp4")
        print(f"[{i}/{len(mp4_files)}] {video_path.name}")

        def progress(frame_num, total_frames):
            if frame_num % 30 == 0 or frame_num == 1:
                print(f"  Frame {frame_num}/{total_frames or '?'}")

        reproject_video(
            video_path=video_path,
            island_metadata=REGION,
            output_path=output_path,
            canvas_width=1920,
            canvas_height=1080,
            crf=18,
            codec="h264_videotoolbox",
            progress_callback=progress,
        )
        print(f"  -> {output_path.name}\n")

    print("Done!")


if __name__ == "__main__":
    main()
