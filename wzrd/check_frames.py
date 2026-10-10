"""
Numeric checks for projector content — no pixels, a compact verdict.

Answers "is this clip safe and clean to put on the wall?" before a human or
an agent spends attention on it:

- **flash** — photosensitive flash rate. A *transition* is a luminance swing
  of at least 10 % of full scale (darker state below 0.8, the WCAG general-
  flash rule) happening at once over at least ``flash_area`` of the frame; a
  flash is a pair of opposing transitions. More than ``max_flashes_per_s`` in
  any one-second window fails. Stricter than broadcast rules on purpose: the
  audience lies on its back staring at a ceiling-sized image.
- **loop_seam** — the last → first frame jump relative to an ordinary frame
  step (a clip that loops cleanly scores ~1).
- **haze** — the dark floor. Projection is additive: anything above black
  lights the surface, so a lifted floor washes the whole wall.
- **lit** — share of the frame carrying real light (informational).
- **shift** — a still (or a video's first/last frame) against the surface
  it was made on: residual misalignment in px, from an affine fit to tile
  phase correlations. Fails above ``shift_tolerance`` of the long side
  (0.1 % ≈ 1.4 px at 1376 — the measured drift was 0.78 %).

Usage::

    python -m wzrd.check_frames clip.mp4
    python -m wzrd.check_frames keyframe.png --surface surface.webp
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .subtract_video import get_video_info
from .utils import align_to_reference

PathLike = Union[str, Path]

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}

# Defaults (each overridable per call)
ANALYSIS_WIDTH = 160          # frames are judged downscaled to this width
FLASH_DELTA = 0.10            # luminance swing that counts (linear, 0-1)
FLASH_DARK_MAX = 0.80         # the darker state must be below this
FLASH_AREA = 0.10             # share of the frame that must swing together
MAX_FLASHES_PER_S = 3.0
SEAM_PASS, SEAM_WARN = 3.0, 6.0
HAZE_FLOOR_WARN = 10          # p10 luma (0-255) above this = lifted black
LIT_THRESHOLD = 40            # luma above this counts as lit
SHIFT_TOLERANCE = 0.001       # fraction of the long side


# ──────────────────────────────────────────────────────────────────────────────
# Frame access
# ──────────────────────────────────────────────────────────────────────────────

def _video_frames(path: PathLike, width: int = ANALYSIS_WIDTH) -> tuple[np.ndarray, float]:
    """All frames as uint8 RGB (T, H, W, 3) at *width*, plus fps."""
    w0, h0, fps, _ = get_video_info(path)
    h = max(2, round(width * h0 / w0 / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={width}:{h}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8)
    n = frames.size // (width * h * 3)
    return frames[: n * width * h * 3].reshape(n, h, width, 3), fps


def _full_frame(path: PathLike, index: int) -> Image.Image:
    """One full-resolution frame (index -1 = last) as a PIL image."""
    cap = cv2.VideoCapture(str(path))
    try:
        if index < 0:
            index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + index
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"cannot read frame {index} of {path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def _linear_luminance(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 → relative luminance 0-1 (WCAG)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722


def _luma8(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────────

def check_flash(frames: np.ndarray, fps: float, *, delta: float = FLASH_DELTA,
                dark_max: float = FLASH_DARK_MAX, area: float = FLASH_AREA,
                max_per_s: float = MAX_FLASHES_PER_S) -> Dict:
    """Large opposing luminance swings per second (see the module docstring).

    Each frame is compared with the frame ~100 ms later, so a fade or a cut
    spread over a few frames is one swing, and slow morphs are none. A frame
    is "up" ("down") when at least *area* of it brightens (darkens) by
    *delta* with the darker state below *dark_max*; a run of same-direction
    frames is one transition, two opposing transitions are one flash.
    """
    lum = _linear_luminance(frames)
    win = max(1, int(round(fps / 10)))
    if len(lum) <= win:
        return {"status": "pass", "max_flashes_per_s": 0.0, "limit": max_per_s, "transitions": 0}
    a, b = lum[:-win], lum[win:]
    darker = np.minimum(a, b) < dark_max
    up = ((b - a >= delta) & darker).mean(axis=(1, 2))
    down = ((a - b >= delta) & darker).mean(axis=(1, 2))
    sign = np.where(up >= area, 1, np.where(down >= area, -1, 0))
    starts: List[int] = []
    last_sign, last_i = 0, -2
    for i, sg in enumerate(sign):
        if sg == 0:
            continue
        if not (sg == last_sign and i - last_i <= 1):
            starts.append(i)
        last_sign, last_i = sg, i
    window = max(1, int(round(fps)))
    worst = max((sum(1 for s in starts[k:] if s - s0 < window) for k, s0 in enumerate(starts)), default=0)
    flashes = worst / 2.0
    return {
        "status": "fail" if flashes > max_per_s else "pass",
        "max_flashes_per_s": round(flashes, 1),
        "limit": max_per_s,
        "transitions": len(starts),
    }


def check_loop_seam(frames: np.ndarray) -> Dict:
    luma = _luma8(frames)
    steps = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    seam = float(np.abs(luma[-1] - luma[0]).mean())
    typical = float(np.median(steps)) if len(steps) else 0.0
    ratio = seam / max(typical, 0.25)  # floor: a near-static clip with a tiny seam is fine
    status = "pass" if ratio <= SEAM_PASS else "warn" if ratio <= SEAM_WARN else "fail"
    return {"status": status, "seam_vs_step": round(ratio, 2),
            "seam_luma": round(seam, 2), "step_luma": round(typical, 2)}


def check_levels(frames: np.ndarray) -> tuple[Dict, Dict]:
    luma = _luma8(frames)
    floor = float(np.median(np.percentile(luma.reshape(len(luma), -1), 10, axis=1)))
    lit = float((luma > LIT_THRESHOLD).mean())
    haze = {"status": "warn" if floor > HAZE_FLOOR_WARN else "pass",
            "floor_luma": round(floor, 1), "limit": HAZE_FLOOR_WARN}
    return haze, {"fraction": round(lit, 3)}


def _grad(im: Image.Image) -> np.ndarray:
    g = cv2.GaussianBlur(np.asarray(im.convert("L"), np.float32), (0, 0), 2.0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (m.max() + 1e-6)


def shift_px(still: Image.Image, surface: Image.Image, tiles: int = 4) -> Optional[float]:
    """Residual misalignment (max corner displacement, px) of *still* on *surface*.

    The still is first mapped into the surface frame exactly as the pipeline
    does (`align_to_reference`, uncrop — never a stretch), then each tile's
    translation is measured by phase correlation and an affine is fitted.
    Tiles where the model added content (low correlation) drop out. None
    when too few tiles agree to say anything.
    """
    still = align_to_reference(still.convert("RGB"), surface.convert("RGB"))[0]
    t, i = _grad(surface), _grad(still)
    h, w = t.shape
    th, tw = h // tiles, w // tiles
    hann = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    src, dst = [], []
    for r in range(tiles):
        for c in range(tiles):
            a = t[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            b = i[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            (dx, dy), resp = cv2.phaseCorrelate(a, b, hann)
            if resp < 0.05 or abs(dx) > tw / 4 or abs(dy) > th / 4:
                continue
            src.append([(c + 0.5) * tw, (r + 0.5) * th, 1.0])
            dst.append([dx, dy])
    if len(src) < 6:
        return None
    fit, *_ = np.linalg.lstsq(np.array(src), np.array(dst), rcond=None)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64)
    return float(np.abs(corners @ fit).max())


def check_shift(stills: List[Image.Image], surface: Image.Image,
                tolerance: float = SHIFT_TOLERANCE) -> Dict:
    limit = tolerance * max(surface.size)
    values = [shift_px(s, surface) for s in stills]
    known = [v for v in values if v is not None]
    if not known:
        return {"status": "warn", "shift_px": None, "limit_px": round(limit, 2),
                "note": "too little shared structure to measure (re-framed, or not this surface)"}
    worst = max(known)
    return {"status": "fail" if worst > limit else "pass",
            "shift_px": [round(v, 2) if v is not None else None for v in values] if len(values) > 1 else round(worst, 2),
            "limit_px": round(limit, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def check_frames(
    media: PathLike,
    surface: Optional[PathLike] = None,
    *,
    max_flashes_per_s: float = MAX_FLASHES_PER_S,
    flash_area: float = FLASH_AREA,
    shift_tolerance: float = SHIFT_TOLERANCE,
) -> Dict:
    """Run every check that applies to *media* (a video or a still).

    Videos: flash, loop_seam, haze, lit. Stills: haze, lit. With *surface*
    the media is a render *of* the surface (a keyframe, a keyframe video), so
    haze/lit don't apply and shift runs instead — on the still, or on the
    video's first and last frame.
    Returns ``{"verdict": "pass"|"warn"|"fail", "checks": {...}}``; the
    verdict is the worst check status.
    """
    media = Path(media)
    checks: Dict[str, Dict] = {}
    if media.suffix.lower() in VIDEO_EXTS:
        frames, fps = _video_frames(media)
        if len(frames) < 2:
            raise ValueError(f"{media} has fewer than two frames")
        checks["flash"] = check_flash(frames, fps, area=flash_area, max_per_s=max_flashes_per_s)
        checks["loop_seam"] = check_loop_seam(frames)
        if surface is None:
            checks["haze"], checks["lit"] = check_levels(frames)
        info = {"frames": len(frames), "fps": round(fps, 2)}
        if surface is not None:
            stills = [_full_frame(media, 0), _full_frame(media, -1)]
    else:
        img = Image.open(media).convert("RGB")
        small = np.asarray(img.resize((ANALYSIS_WIDTH, max(2, round(ANALYSIS_WIDTH * img.height / img.width)))))
        if surface is None:
            checks["haze"], checks["lit"] = check_levels(small[None])
        info = {"size": list(img.size)}
        stills = [img]
    if surface is not None:
        checks["shift"] = check_shift(stills, Image.open(surface), tolerance=shift_tolerance)
    order = {"pass": 0, "warn": 1, "fail": 2}
    statuses = [c["status"] for c in checks.values() if "status" in c]
    verdict = max(statuses, key=order.__getitem__) if statuses else "pass"
    return {"verdict": verdict, **info, "checks": checks}


def _cli() -> None:
    p = argparse.ArgumentParser(prog="python -m wzrd.check_frames", description=__doc__.split("\n\n")[0])
    p.add_argument("media", help="video or still")
    p.add_argument("--surface", help="surface the media was made on (enables the shift check)")
    p.add_argument("--max-flashes", type=float, default=MAX_FLASHES_PER_S)
    args = p.parse_args()
    print(json.dumps(check_frames(args.media, args.surface, max_flashes_per_s=args.max_flashes), indent=2))


if __name__ == "__main__":
    _cli()
