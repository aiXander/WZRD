"""
Playable clips for the realtime engine: HAP transcode, thumbnails, stills.

The engine plays HAP only (gig D2: all-intra → seamless loops, free seeks,
GPU-native BC textures), so every rendered mp4 that should reach the wall
goes through :func:`to_hap`. The clip-file layout around it (gig contract
C3: ``<stem>.mp4`` → ``<stem>.mov`` + ``<stem>.thumb.jpg``) is the MCP's
job; this module only runs ffmpeg.

Usage::

    python -m wzrd.clips clip.mp4                # → clip.mov + clip.thumb.jpg
    python -m wzrd.clips clip.mp4 --codec hap_q
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

from .subtract_video import get_video_info

PathLike = Union[str, Path]

#: ffmpeg `hap` encoder formats the engine decodes (render-core `video.rs`).
HAP_FORMATS = ("hap", "hap_q", "hap_alpha")
THUMB_WIDTH = 320


def _run(cmd: list) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg failed: {' | '.join(tail)}")


def _duration(path: PathLike) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out or 0.0)


def to_hap(src: PathLike, dst: Optional[PathLike] = None, codec: str = "hap") -> Dict:
    """Transcode a video to a HAP ``.mov`` the engine can play.

    BC textures work in 4x4 blocks and ffmpeg's encoder refuses other sizes,
    so odd dimensions are padded (centred, black — additive projection makes
    black free) up to multiples of 4. Frame rate and frame count are kept;
    audio is dropped.

    Args:
        src: Source video (any ffmpeg-readable format).
        dst: Output path; default ``<src stem>.mov`` next to *src*.
        codec: ``"hap"`` (DXT1, smallest, default), ``"hap_q"`` (better
            colour, ~2x size) or ``"hap_alpha"``.

    Returns:
        ``{"path", "codec", "size": [w, h], "padded": bool, "fps", "frames", "mb"}``.
    """
    if codec not in HAP_FORMATS:
        raise ValueError(f"codec must be one of {HAP_FORMATS}, got {codec!r}")
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".mov")
    w, h, fps, frames = get_video_info(src)
    pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
    vf = ["-vf", f"pad={pw}:{ph}:({pw}-iw)/2:({ph}-ih)/2:black"] if (pw, ph) != (w, h) else []
    # Temp file + rename: a reader (the engine playing the old clip) keeps its
    # inode and never sees a half-written file.
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    try:
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *vf, "-an",
              "-c:v", "hap", "-format", codec, str(tmp)])
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "path": str(dst), "codec": codec, "size": [pw, ph], "padded": (pw, ph) != (w, h),
        "fps": round(fps, 3), "frames": frames, "mb": round(dst.stat().st_size / 1e6, 1),
    }


def thumbnail(src: PathLike, dst: PathLike, width: int = THUMB_WIDTH) -> str:
    """The middle frame of *src* as a JPEG *width* px wide."""
    t = _duration(src) / 2
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src),
          "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(dst)])
    return str(dst)


def extract_frame(src: PathLike, at: Union[float, str], dst: PathLike) -> str:
    """One frame as a still: *at* = seconds, ``"first"`` or ``"last"``.

    ``"last"`` decodes the final second and keeps the last frame written,
    which is exact for any frame rate (a seek to the duration is not).
    """
    if at == "first":
        cmd = ["-i", str(src), "-frames:v", "1"]
    elif at == "last":
        cmd = ["-sseof", "-1", "-i", str(src), "-update", "1"]
    else:
        cmd = ["-ss", f"{float(at):.3f}", "-i", str(src), "-frames:v", "1"]
    _run(["ffmpeg", "-y", "-v", "error", *cmd, str(dst)])
    if not Path(dst).is_file():
        raise RuntimeError(f"no frame at {at!r} in {src}")
    return str(dst)


def _cli() -> None:
    p = argparse.ArgumentParser(prog="python -m wzrd.clips", description="Video → HAP .mov + thumbnail.")
    p.add_argument("video")
    p.add_argument("--codec", default="hap", choices=HAP_FORMATS)
    args = p.parse_args()
    info = to_hap(args.video, codec=args.codec)
    info["thumb"] = thumbnail(args.video, Path(info["path"]).with_suffix(".thumb.jpg"))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    _cli()
