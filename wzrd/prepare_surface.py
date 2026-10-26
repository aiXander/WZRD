"""
Surface rectification for projection mapping: the show-setup surface step.

`rectify_surface` turns the lit photo (projector showing full white, shot from
the projector's position) plus an optional ambient photo (same view, room lit)
into the show's three surface images at the canvas: the rectified lit crop,
the dark colour-corrected surface, and the aligned ambient photo.

Example usage:
    python -m wzrd.prepare_surface lit.jpg ambient.jpg --size 1376x768 -o surfaces/
"""

import cv2
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from PIL import Image

from .detect import detect_projection_area
from .align import align_images
from .darken import (
    darken_image,
    MAX_BRIGHTNESS,
    DETAIL_BOOST,
    GUIDE_FRACTION,
    CHROMA_CORRECTION,
    GRADIENT_SOURCE,
)


def rectify_surface(
    lit_image_path: Union[str, Path],
    ambient_image_path: Optional[Union[str, Path]] = None,
    *,
    output_size: Tuple[int, int],
    corners: Optional[Sequence[Sequence[float]]] = None,
    margin: float = 0.01,
    min_projection_fraction: float = 0.10,
    max_brightness: float = MAX_BRIGHTNESS,
    detail_boost: float = DETAIL_BOOST,
    guide_fraction: float = GUIDE_FRACTION,
    chroma_correction: float = CHROMA_CORRECTION,
    gradient_source: float = GRADIENT_SOURCE,
) -> dict:
    """
    Lit photo (+ optional ambient photo) -> the three surface images of a show.

    1. Rectify the projector's lit quad in *lit_image_path* to exactly
       *output_size* (the canvas): auto-detected, or the manual normalised
       *corners* (TL, TR, BR, BL) used as given (no margin).
    2. With an ambient photo (same view, room lit): align it to the lit crop
       and darken that. Without one: darken the lit crop.

    Returns a dict:
        lit      — PIL RGB, the rectified lit crop (shape/segmentation input)
        surface  — PIL RGB, the dark colour-corrected surface
        ambient  — PIL RGB, the aligned undarkened ambient photo, or None
        corners  — the quad actually rectified, normalised (round-trips as
                   `corners=`)
        manual_corners — bool
        align    — {method, confidence, num_inliers, num_matches} or None
        size     — (w, h)
    """

    w, h = int(output_size[0]), int(output_size[1])
    lit_bgr, info = detect_projection_area(
        lit_image_path,
        margin=margin,
        min_projection_fraction=min_projection_fraction,
        output_size=(w, h),
        corners=corners,
    )

    align_info = None
    ambient_rgb = None
    if ambient_image_path is not None:
        ambient_bgr = cv2.imread(str(ambient_image_path))
        if ambient_bgr is None:
            raise FileNotFoundError(f"Could not load ambient image: {ambient_image_path}")
        aligned_bgr, res = align_images(ambient_bgr, lit_bgr)
        ambient_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        align_info = {
            'method': res.method,
            'confidence': round(float(res.confidence), 4),
            'num_inliers': int(res.num_inliers),
            'num_matches': int(res.num_matches),
        }

    lit_rgb = cv2.cvtColor(lit_bgr, cv2.COLOR_BGR2RGB)
    base = ambient_rgb if ambient_rgb is not None else lit_rgb
    dark = darken_image(
        base,
        max_brightness=max_brightness,
        detail_boost=detail_boost,
        guide_fraction=guide_fraction,
        chroma_correction=chroma_correction,
        gradient_source=gradient_source,
    )
    return {
        'lit': Image.fromarray(lit_rgb),
        'surface': Image.fromarray(dark),
        'ambient': Image.fromarray(ambient_rgb) if ambient_rgb is not None else None,
        'corners': [[round(float(x), 6), round(float(y), 6)] for x, y in info['corners_crop']],
        'manual_corners': bool(info['manual_corners']),
        'align': align_info,
        'size': (w, h),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    """CLI entry point for ``python -m wzrd.prepare_surface``."""
    import argparse

    parser = argparse.ArgumentParser(description='Lit (+ ambient) photo -> the rectified lit crop, the '
                                                 'dark surface and the aligned ambient photo.')
    parser.add_argument('lit', help='Photo of the projector showing full white, from its position')
    parser.add_argument('ambient', nargs='?', default=None, help='Same view with the room lit (optional)')
    parser.add_argument('--size', required=True, help='Output canvas, WxH (e.g. 1376x768)')
    parser.add_argument('-o', '--output-dir', default='.', help='Folder for <stem>_{lit,surface,ambient}.png')
    parser.add_argument('--margin', type=float, default=0.01, help='Shrink the detected quad inward (default 0.01)')
    parser.add_argument('--max-brightness', type=float, default=MAX_BRIGHTNESS,
                        help=f'Luminance budget ceiling (default {MAX_BRIGHTNESS})')
    args = parser.parse_args()

    w, h = (int(v) for v in args.size.lower().split('x'))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = rectify_surface(args.lit, args.ambient, output_size=(w, h),
                          margin=args.margin, max_brightness=args.max_brightness)
    stem = Path(args.lit).stem
    for key in ('lit', 'surface', 'ambient'):
        if res[key] is not None:
            path = out / f'{stem}_{key}.png'
            res[key].save(path)
            print(f'{key}: {path}')
    print(f"corners: {res['corners']}")
    return 0


if __name__ == '__main__':
    exit(_main())
