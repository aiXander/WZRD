"""Stitch video segments into one clip — e.g. keyframe-to-keyframe segments
into a continuous (optionally seamlessly looping) story.

Image→video models render a start→end segment whose first frame is the start
keyframe and whose last frame is the end keyframe. Played back to back, each
shared keyframe would show twice (a visible hitch), so by default the first
frame of every segment after the first is dropped. With `seamless_loop`, the
very last frame is dropped as well: in a closed loop (last segment ends on the
first keyframe) it duplicates frame 0 on wrap-around.

All segments are conformed to the first segment's size and frame rate.

CLI: python -m wzrd.concat seg1.mp4 seg2.mp4 ... -o story.mp4 [--loop]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2

from .utils import VideoWriter

PathLike = Union[str, Path]


def concat_videos(
    paths: Sequence[PathLike],
    output_path: PathLike,
    *,
    drop_boundary_frames: bool = True,
    seamless_loop: bool = False,
    fps: Optional[float] = None,
    crf: int = 18,
    codec: str = "libx264",
) -> dict:
    """Concatenate `paths` in order into `output_path` (mp4). Returns stats."""
    if not paths:
        raise ValueError("concat_videos needs at least one video")

    first = cv2.VideoCapture(str(paths[0]))
    if not first.isOpened():
        raise ValueError(f"cannot open video {paths[0]}")
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or first.get(cv2.CAP_PROP_FPS) or 24.0
    first.release()

    frames = 0
    pending = None  # one-frame delay, so the final frame can be dropped for loops
    with VideoWriter(output_path, width, height, fps, crf=crf, codec=codec) as writer:
        for i, path in enumerate(paths):
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"cannot open video {path}")
            skip_first = drop_boundary_frames and i > 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if skip_first:
                    skip_first = False
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if pending is not None:
                    writer.write(pending)
                    frames += 1
                pending = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
        if pending is not None and not seamless_loop:
            writer.write(pending)
            frames += 1

    return {
        "frames": frames,
        "fps": fps,
        "size": [width - width % 2, height - height % 2],
        "duration_s": round(frames / fps, 3),
        "segments": len(paths),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--loop", action="store_true", help="drop the final frame for a seamless loop")
    parser.add_argument("--keep-boundary-frames", action="store_true")
    parser.add_argument("--codec", default="libx264")
    args = parser.parse_args()
    info = concat_videos(
        args.videos, args.output, seamless_loop=args.loop,
        drop_boundary_frames=not args.keep_boundary_frames, codec=args.codec,
    )
    print(info)


if __name__ == "__main__":
    _cli()
