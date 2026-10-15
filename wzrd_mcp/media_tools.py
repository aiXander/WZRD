"""Media tools: let the agent SEE what it made, and stitch segments together.

`view_media` closes the loop the local-first move opened: tools return paths,
and a client without file access (Claude Desktop) would otherwise never look
at a keyframe before choosing it. Images come back downscaled; videos as one
contact sheet of evenly spaced frames — enough to catch camera drift, a
changed background or a dark subject, at a fraction of the tokens.
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Optional, Union

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image

from . import library, project
from .file_io import resolve_input_async
from .server import tool

_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
_MAX_ITEMS = 6


def _jpeg(img, max_side: int) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def _image_preview(path: str, max_side: int) -> tuple[str, bytes]:
    from PIL import Image as PILImage

    with PILImage.open(path) as img:
        caption = f"{path} — image {img.width}x{img.height}"
        return caption, _jpeg(img, max_side)


def _video_sheet(path: str, max_side: int, frames: int) -> tuple[str, bytes]:
    """Evenly spaced frames tiled into one grid, each stamped with its time."""
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ToolError(f"cannot open video {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = max(1, min(frames, total))
        picks = [round(i * (total - 1) / max(1, n - 1)) for i in range(n)]
        tiles = []
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.putText(frame, f"{idx / fps:.1f}s", (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.8, h / 600), (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not tiles:
        raise ToolError(f"no decodable frames in {path}")
    cols = 3 if len(tiles) > 4 else 2 if len(tiles) > 1 else 1
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = t[:h, :w]
    caption = (f"{path} — video {w}x{h}, {total / fps:.1f}s @ {fps:.0f} fps; "
               f"contact sheet of {len(tiles)} frames, left→right, top→bottom")
    return caption, _jpeg(PILImage.fromarray(sheet), max_side * 2)


@tool(title="View images / videos", tags={"preview"}, read_only=True, idempotent=True)
async def view_media(
    paths: list[str],
    max_side: int = 768,
    video_frames: int = 6,
    ctx: Optional[Context] = None,
) -> list:
    """Look at images and videos: returns downscaled images, and one contact sheet of evenly spaced frames per video.

    Use it to check generated keyframes and clips before presenting or
    building on them: is the background unchanged, the subject bright and
    the right size, the camera static, the motion plausible?

    Args:
        paths: Up to 6 local paths or URLs (images or videos).
        max_side: Long side in pixels per image (videos get a sheet up to 2x this). Smaller = fewer tokens.
        video_frames: Frames per video contact sheet (1-12).
    """
    if not paths:
        raise ToolError("paths is empty")
    if len(paths) > _MAX_ITEMS:
        raise ToolError(f"at most {_MAX_ITEMS} items per call (got {len(paths)}); split the call")
    max_side = max(128, min(max_side, 1568))
    video_frames = max(1, min(video_frames, 12))

    async def one(ref: str) -> list:
        local = await resolve_input_async(ref)
        is_video = Path(local).suffix.lower() in _VIDEO_EXTS
        caption, data = await asyncio.to_thread(
            _video_sheet if is_video else _image_preview,
            local, max_side, *([video_frames] if is_video else []),
        )
        return [caption, Image(data=data, format="jpeg")]

    out: list = []
    for item in await asyncio.gather(*(one(p) for p in paths)):
        out += item
    return out


@tool(title="Concatenate videos", tags={"content", "video"}, dest_arg=True)
async def concat_videos(
    videos: list[str],
    seamless_loop: bool = False,
    drop_boundary_frames: bool = True,
    codec: str = "libx264",
    crf: int = 18,
    ctx: Optional[Context] = None,
) -> dict:
    """Stitch video segments, in order, into one clip — e.g. keyframe→keyframe segments into the full story.

    Segments are conformed to the first one's size and fps. By default the
    first frame of each later segment is dropped (it repeats the previous
    segment's end keyframe).

    Args:
        videos: Segment paths/URLs in playback order (2+).
        seamless_loop: The last segment ends on the first keyframe: also drop the final frame so the clip loops without a hitch.
        drop_boundary_frames: Drop each later segment's first frame (the shared keyframe). Turn off for unrelated clips.
        codec: ffmpeg video codec (libx264 everywhere; h264_videotoolbox on a Mac is faster).
        crf: Quality — lower is better (0-51).
    """
    from wzrd.concat import concat_videos as _concat

    if len(videos) < 2:
        raise ToolError("concat_videos needs at least 2 videos")
    local = await asyncio.gather(*(resolve_input_async(v, suffix=".mp4") for v in videos))
    out_path = library.output_path("concat_videos", ".mp4")
    info = await asyncio.to_thread(
        _concat, local, out_path,
        drop_boundary_frames=drop_boundary_frames, seamless_loop=seamless_loop, crf=crf, codec=codec,
    )
    return {"output_video": out_path, "info": info}


@tool(title="Check frames", tags={"preview", "qc"}, read_only=True, idempotent=True)
async def check_frames(
    media: str,
    surface: Optional[str] = None,
    max_flashes_per_s: float = 3.0,
    ctx: Optional[Context] = None,
) -> dict:
    """Numeric QC of a clip or still before it goes on the wall — compact JSON, no pixels.

    Video: `flash` (photosensitive: > max_flashes_per_s large flashes in any
    second fails — the audience lies staring at the ceiling), `loop_seam`
    (last → first jump vs a normal frame step; ~1 = clean loop), `haze`
    (lifted black floor washes the whole surface), `lit` (share of frame
    lit). Still: haze + lit. With `surface`: the media is a render OF that
    surface (keyframe, keyframe video) and `shift` measures its residual
    misalignment in px (fails above 0.1 % of the long side). `verdict` is the
    worst status (pass < warn < fail). Run it on every new clip; never put a
    flash fail on the wall.

    Args:
        media: Path or URL of a video or image.
        surface: Path or URL of the surface the media was made on (enables the shift check).
        max_flashes_per_s: Flash limit (default 3, the photosensitive standard). Lower it for a quieter set; never raise it for a live audience.
    """
    from wzrd.check_frames import check_frames as _check

    media_path = await resolve_input_async(media)
    surface_path = await resolve_input_async(surface) if surface else None
    result = await asyncio.to_thread(_check, media_path, surface_path, max_flashes_per_s=max_flashes_per_s)
    return {"media": media_path, **result}


# ---------------------------------------------------------------------------
# Playable clips (Studio S8): <stem>.mp4 -> <stem>.mov (HAP) beside it, same render id
# ---------------------------------------------------------------------------
def _beside_ok(src: Path) -> bool:
    """A HAP may go next to its source: a real folder we can write, not scratch."""
    import tempfile

    tmp = Path(tempfile.gettempdir()).resolve()
    return os.access(src.parent, os.W_OK) and not src.is_relative_to(tmp) and \
        not src.is_relative_to(library.wzrd_dir())


def _same_render(a: Path, b: Path) -> bool:
    """Is *a* already a file of *b*'s render (an earlier HAP of it — replaced, not duplicated)?"""
    try:
        return library.resolve_id(str(a)) == library.resolve_id(str(b))
    except ValueError:
        return False


def make_clip(video: str, codec: str = "hap") -> dict:
    """Transcode *video* to a HAP `.mov` the engine plays (blocking).

    The clip goes beside its source with the same stem (S8: the mp4 stays the
    master, the HAP is a disposable cache); a source in a temp folder (a
    downloaded URL) or an unwritable one sends it to the library instead.
    """
    from wzrd.clips import to_hap

    src = Path(video).resolve()
    if _beside_ok(src):
        mov = src.with_suffix(".mov") if src.suffix.lower() != ".mov" else src.with_name(f"{src.stem}_hap.mov")
        if mov.exists() and not _same_render(mov, src):
            mov = project.unique(src.parent, mov.stem, ".mov")  # a foreign file of that name stays
    else:
        mov = Path(library.output_path("to_hap", ".mov", name=src.stem))
    info = to_hap(src, mov, codec)
    return {"clip": str(mov), "hap": {k: info[k] for k in ("codec", "size", "padded", "mb")}}


async def attach_clip(result: dict, video_key: str, hap: Union[bool, str]) -> dict:
    """After a render tool downloads its mp4: add the HAP clip to *result*,
    recorded as a variant of the mp4 (same render id). `hap` = False, True
    ("hap") or a codec name."""
    if not hap or not result.get(video_key):
        return result
    codec = hap if isinstance(hap, str) else "hap"
    clip = await asyncio.to_thread(make_clip, result[video_key], codec)
    result.update(clip)
    prov = result.setdefault("_provenance", {})
    prov.setdefault("variant_of", {})[clip["clip"]] = str(Path(result[video_key]).resolve())
    return result


@tool(title="Video to HAP clip", tags={"content", "engine"}, idempotent=True, project_arg=True, render=True)
async def to_hap(video: str, codec: str = "hap", region: Optional[str] = None,
                 ctx: Optional[Context] = None) -> dict:
    """Make a video playable by the engine: a HAP .mov beside the source (same stem, same render id — labels are shared).

    texture_flow and kling_* already do this for their own renders
    (hap=True); use this for any other video, a DXV from the Resolume
    library included. Dimensions are padded to multiples of 4, fps kept.
    HAP is 5-10x bigger than H.264: the mp4 stays the master, the HAP is a
    disposable cache. A URL source lands in the library (dest=).

    Args:
        video: Path or URL of the source video.
        codec: "hap" (default, smallest), "hap_q" (better colour, ~2x size) or "hap_alpha".
        region: Pack layer id the clip is for, when the source records none (a stitched or subtracted video).
    """
    from .provenance import region_meta

    info = region_meta(region) if region else {}  # refuse an unknown id before transcoding
    path = await resolve_input_async(video, suffix=".mp4")
    result = await asyncio.to_thread(make_clip, path, codec)
    result["_provenance"] = {**info, "variant_of": {result["clip"]: path}}
    return {"video": path, **result}


@tool(title="Extract frames", tags={"content"}, dest_arg=True)
async def extract_frames(
    video: str,
    at: Optional[list[Union[float, str]]] = None,
    count: Optional[int] = None,
    ctx: Optional[Context] = None,
) -> dict:
    """Pull stills out of a video into the media library — e.g. a clip's last frame as the next keyframe or start frame.

    Default: the first and last frame. Stills are PNG (dest= picks the folder).

    Args:
        video: Path or URL of the video.
        at: Which frames: seconds (e.g. 2.5) and/or "first" / "last".
        count: Instead of `at`: N evenly spaced frames from first to last (2-24).
    """
    from wzrd.clips import _duration, extract_frame

    path = await resolve_input_async(video, suffix=".mp4")
    if count is not None:
        if not 2 <= count <= 24:
            raise ToolError("count must be 2-24")
        dur = await asyncio.to_thread(_duration, path)
        at = ["first", *[round(dur * i / (count - 1), 3) for i in range(1, count - 1)], "last"]
    at = at or ["first", "last"]
    stem = Path(path).stem

    def one(t: Union[float, str]) -> dict:
        label = t if isinstance(t, str) else f"{float(t):.2f}s".replace(".", "_")
        out = library.output_path("extract_frames", ".png", name=f"{stem}_{label}")
        return {"at": t, "path": extract_frame(path, t, out)}

    frames = [await asyncio.to_thread(one, t) for t in at]
    return {"video": path, "frames": frames}
