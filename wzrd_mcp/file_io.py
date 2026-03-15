"""File I/O: download URLs to local tmp files, upload local files to S3."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import httpx

# ---------------------------------------------------------------------------
# S3 client (lazy singleton)
# ---------------------------------------------------------------------------
_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.getenv("AWS_REGION_NAME", "us-east-1"),
        )
    return _s3_client


def _bucket():
    return os.environ["AWS_BUCKET_NAME"]


def _cdn_root() -> str | None:
    return os.getenv("CLOUDFRONT_URL")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_input(url_or_path: str, suffix: str = "") -> str:
    """URL → download to tmp file.  Base64 → decode to tmp file.  Local path → validate and return."""
    if url_or_path.startswith(("http://", "https://")):
        return _download(url_or_path, suffix)
    if url_or_path.startswith("base64:"):
        return _decode_base64(url_or_path[7:], suffix)
    if url_or_path.startswith("data:"):
        # data:image/png;base64,iVBOR...
        _, encoded = url_or_path.split(",", 1)
        return _decode_base64(encoded, suffix)
    path = Path(url_or_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return str(path)


def upload(local_path: str) -> str:
    """Upload a local file to S3. Returns the public URL (CDN if configured, else S3)."""
    path = Path(local_path)
    data = path.read_bytes()

    # Content-hash name for deduplication (same scheme as eve)
    name = hashlib.sha256(data).hexdigest()
    ext = path.suffix or _guess_ext(data)
    key = f"{name}{ext}"

    bucket = _bucket()
    s3 = _s3()

    # Skip upload if already exists
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
            s3.upload_fileobj(
                path.open("rb"),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type, "ContentDisposition": "inline"},
            )
        else:
            raise

    # cdn = _cdn_root()
    # if cdn:
    #     return f"{cdn}/{key}"
    region = os.getenv("AWS_REGION_NAME", "us-east-1")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def make_temp_path(suffix: str = "", prefix: str = "wzrd_") -> str:
    """Create a temp file path (file not yet written)."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    return path


def make_temp_dir(prefix: str = "wzrd_") -> str:
    """Create a temp directory."""
    return tempfile.mkdtemp(prefix=prefix)


# ---------------------------------------------------------------------------
# Async wrappers (offload blocking I/O to threads)
# ---------------------------------------------------------------------------


async def resolve_input_async(url_or_path: str, suffix: str = "") -> str:
    """Async version of resolve_input — offloads downloads/decoding to a thread."""
    return await asyncio.to_thread(resolve_input, url_or_path, suffix)


async def upload_async(local_path: str) -> str:
    """Async version of upload — offloads S3 I/O to a thread."""
    return await asyncio.to_thread(upload, local_path)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _decode_base64(encoded: str, suffix: str = "") -> str:
    data = base64.b64decode(encoded)
    if not suffix:
        suffix = _guess_ext(data)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _download(url: str, suffix: str = "") -> str:
    if not suffix:
        url_path = urlparse(url).path
        if "." in Path(url_path).name:
            suffix = Path(url_path).suffix

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        resp = client.get(url)
        resp.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def _guess_ext(data: bytes) -> str:
    """Guess file extension from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[4:12] == b"ftypmp4" or data[4:12] == b"ftypisom":
        return ".mp4"
    return ""
