"""Local-only MCP tools — hardware access that only works on localhost."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project
from ._log import log_call, log_progress, log_done, log_error, logged_tool
from .server import mcp, get_timeout


def _capture_frame(out_path: str, camera_index: int = 0, warmup_frames: int = 5) -> str:
    """Capture a single frame from a camera via OpenCV. Returns the saved path."""
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}. "
            "Check that a camera is connected and not in use by another app."
        )
    try:
        # Discard initial frames so auto-exposure/white-balance can settle
        for _ in range(warmup_frames):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Camera opened but failed to capture a frame.")

        cv2.imwrite(out_path, frame, [cv2.IMWRITE_WEBP_QUALITY, 90])
        return out_path
    finally:
        cap.release()


@mcp.tool(timeout=get_timeout("capture_camera_snapshot"))
@logged_tool
async def capture_camera_snapshot(
    camera_index: int = 0,
    ctx: Optional[Context] = None,
) -> dict:
    """Capture a snapshot from a connected camera (webcam, USB camera, etc.).

    This tool only works when the MCP server is running locally (not on Modal).
    The frame is saved into the active project's `captures/` folder and its
    local path is returned.

    Args:
        camera_index: Which camera to use (0 = default/built-in, 1+ = external USB cameras).
    """
    _name = "capture_camera_snapshot"
    t0 = time.time()
    try:
        out_path = project.output_path(
            f"cam{camera_index}", ".webp", kind=project.CAPTURES
        )

        log_progress(_name, f"Capturing from camera {camera_index}...")
        img_path = await asyncio.to_thread(_capture_frame, out_path, camera_index)

        result = {
            "snapshot_image": img_path,
            "info": {
                "camera_index": camera_index,
            },
        }
        log_done(_name, t0, result)
        return result
    except ToolError:
        raise
    except Exception as exc:
        log_error(_name, exc, t0)
        raise ToolError(str(exc))
