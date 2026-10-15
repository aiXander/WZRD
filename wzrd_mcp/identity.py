"""A media file's identity, read off the file itself (Studio contracts L2, L3).

Pure file work, no database: the content **fingerprint**, the embedded `wzrd`
tag (read + write), a probe (codec, size, frames) and the cache thumbnail.
`library.py` decides what the results mean.

- **Fingerprint** = ``"<size>-<sha1[:20]>"``, the sha1 over the first and the
  last 1 MiB. It identifies a file that carries no tag (a DXV, a style image,
  a webp) wherever it moves; a path is never an identity.
- **The `wzrd` tag** holds the render's creation record as JSON, written once
  at creation and never rewritten:

  | Container | Where |
  |---|---|
  | mp4 / mov | format-level metadata key `wzrd` (`ffmpeg -map 0 -c copy -movflags use_metadata_tags`, temp file + rename) |
  | png | an iTXt chunk with keyword `wzrd`, before `IEND` |
  | jpg | a COM segment whose payload is `wzrd=<json>`, after the APPn segments |

  Other formats (webp, gif, webm, heic) carry no tag; the `files` row's
  fingerprint links them to their record instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any, Optional

TAG_KEY = "wzrd"
TAG_SCHEMA = 1
_JPEG_PREFIX = b"wzrd="
_CHUNK = 1 << 20  # 1 MiB from each end
THUMB_WIDTH = 320

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
TAGGABLE_EXTS = {".mp4", ".mov", ".m4v", ".png", ".jpg", ".jpeg"}


def media_kind(path: str | Path) -> Optional[str]:
    ext = Path(path).suffix.lower()
    return "video" if ext in VIDEO_EXTS else "image" if ext in IMAGE_EXTS else None


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def fingerprint(path: str | Path) -> str:
    """``"<size>-<sha1 of the first + last 1 MiB, 20 hex>"`` — cheap at any size."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1()
    with p.open("rb") as fh:
        h.update(fh.read(_CHUNK))
        if size > _CHUNK:
            fh.seek(max(_CHUNK, size - _CHUNK))
            h.update(fh.read(_CHUNK))
    return f"{size}-{h.hexdigest()[:20]}"


# ---------------------------------------------------------------------------
# The embedded tag
# ---------------------------------------------------------------------------
def _tmp_beside(path: Path) -> Path:
    return path.with_name(f".{path.stem}.wzrdtag{path.suffix}")


def write_tag(path: str | Path, record: dict) -> bool:
    """Embed *record* as the file's `wzrd` tag. False for an untaggable format.

    Written into a temp file beside the original and renamed over it, so a
    reader never sees a half-written file.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in TAGGABLE_EXTS:
        return False
    text = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
    tmp = _tmp_beside(p)
    try:
        if ext in (".mp4", ".mov", ".m4v"):
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(p), "-map", "0", "-c", "copy",
                 "-movflags", "use_metadata_tags", "-metadata", f"{TAG_KEY}={text}", str(tmp)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg tag write failed: {proc.stderr.strip()[-300:]}")
        elif ext == ".png":
            tmp.write_bytes(_png_with_tag(p.read_bytes(), text))
        else:
            tmp.write_bytes(_jpeg_with_tag(p.read_bytes(), text))
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _png_with_tag(data: bytes, text: str) -> bytes:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    iend = data.rfind(b"IEND") - 4
    itxt = TAG_KEY.encode() + b"\x00\x00\x00\x00\x00" + text.encode("utf-8")
    return data[:iend] + _png_chunk(b"iTXt", itxt) + data[iend:]


def _jpeg_with_tag(data: bytes, text: str) -> bytes:
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    payload = _JPEG_PREFIX + text.encode("utf-8")
    if len(payload) > 65533:
        raise ValueError("record too large for a JPEG comment")
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF and 0xE0 <= data[i + 1] <= 0xEF:  # skip APPn
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return data[:i] + b"\xff\xfe" + struct.pack(">H", len(payload) + 2) + payload + data[i:]


def _png_tag(path: Path) -> Optional[str]:
    with path.open("rb") as fh:
        if fh.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return None
            length, kind = struct.unpack(">I", head[:4])[0], head[4:]
            if kind == b"iTXt":
                body = fh.read(length)
                if body.startswith(TAG_KEY.encode() + b"\x00"):
                    # keyword\0 flag method lang\0 translated\0 text
                    rest = body[len(TAG_KEY) + 3:]
                    rest = rest.split(b"\x00", 2)[-1]
                    return rest.decode("utf-8")
                fh.seek(4, 1)
            elif kind == b"IEND":
                return None
            else:
                fh.seek(length + 4, 1)


def _jpeg_tag(path: Path) -> Optional[str]:
    with path.open("rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            return None
        while True:
            head = fh.read(4)
            if len(head) < 4 or head[0] != 0xFF:
                return None
            marker, length = head[1], struct.unpack(">H", head[2:])[0]
            if marker == 0xDA:  # start of scan: no more headers
                return None
            body = fh.read(length - 2)
            if marker == 0xFE and body.startswith(_JPEG_PREFIX):
                return body[len(_JPEG_PREFIX):].decode("utf-8")


def _parse(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    try:
        tag = json.loads(text)
    except ValueError:
        return None
    return tag if isinstance(tag, dict) and str(tag.get("id", "")).startswith(("r_", "fp_")) else None


def read_tag(path: str | Path, probe: Optional[dict] = None) -> Optional[dict]:
    """The file's `wzrd` record, or None. Videos read it from *probe* if given."""
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext == ".png":
            return _parse(_png_tag(p))
        if ext in (".jpg", ".jpeg"):
            return _parse(_jpeg_tag(p))
        if ext in VIDEO_EXTS:
            probe = probe if probe is not None else probe_file(p)
            return _parse((probe or {}).get("_tag"))
    except (OSError, ValueError, struct.error):
        return None
    return None


# ---------------------------------------------------------------------------
# Probe + thumbnail
# ---------------------------------------------------------------------------
def probe_file(path: str | Path) -> dict:
    """`{kind, codec, w, h, frames, fps, duration_s}` (+ `_tag` for a video). Empty on failure."""
    p = Path(path)
    kind = media_kind(p)
    if kind == "image":
        try:
            from wzrd.utils import _register_heif

            _register_heif()
            from PIL import Image

            with Image.open(p) as im:
                return {"kind": "image", "codec": (im.format or p.suffix[1:]).lower(),
                        "w": im.width, "h": im.height, "frames": getattr(im, "n_frames", 1),
                        "fps": None, "duration_s": None}
        except Exception:  # noqa: BLE001 — an unreadable file is still listed
            return {"kind": "image"}
    if kind != "video":
        return {}
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,codec_tag_string,width,height,nb_frames,r_frame_rate"
         ":format=duration:format_tags=" + TAG_KEY, "-of", "json", str(p)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"kind": "video"}
    info = json.loads(proc.stdout or "{}")
    st = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    num, _, den = str(st.get("r_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else None
    dur = float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None
    frames = int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else (
        round(dur * fps) if dur and fps else None)
    return {"kind": "video", "codec": st.get("codec_name"), "w": st.get("width"), "h": st.get("height"),
            "frames": frames, "fps": round(fps, 3) if fps else None,
            "duration_s": round(dur, 3) if dur else None,
            "_tag": (fmt.get("tags") or {}).get(TAG_KEY)}


def make_thumb(path: str | Path, dest: str | Path, duration_s: Optional[float] = None,
               width: int = THUMB_WIDTH) -> bool:
    """A JPEG *width* px wide: an image's downscale or a video's middle frame."""
    p, dest = Path(path), Path(dest)
    tmp = dest.with_name(f".{dest.name}")
    try:
        if media_kind(p) == "image":
            from wzrd.utils import load_upright

            img = load_upright(p).convert("RGB")
            img.thumbnail((width, width * 4))
            img.save(tmp, "JPEG", quality=82)
        else:
            t = (duration_s or 0) / 2
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(p), "-frames:v", "1",
                 "-vf", f"scale={width}:-2", "-q:v", "4", "-f", "image2", str(tmp)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0 or not tmp.is_file():
                return False
        os.replace(tmp, dest)
        return True
    except Exception:  # noqa: BLE001 — a missing thumb is a placeholder, not an error
        tmp.unlink(missing_ok=True)
        return False


def new_id(prefix: str = "r_") -> str:
    """A ULID (time-ordered, Crockford base32) with *prefix*: ``r_01J…``."""
    import secrets
    import time

    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    n = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    out = []
    for _ in range(26):
        n, r = divmod(n, 32)
        out.append(alphabet[r])
    return prefix + "".join(reversed(out))


def strip_private(probe: dict) -> dict[str, Any]:
    return {k: v for k, v in probe.items() if not k.startswith("_")}
