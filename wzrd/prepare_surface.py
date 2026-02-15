"""
Surface preparation pipeline for projection mapping.

Combines detect → align → darken into a single entrypoint that produces
a darkened reference/background image of the projection surface.

Two modes:

1. **Full pipeline** (day photo + night photo):
   - detect:  find the projection quadrilateral in the night image
   - align:   warp the day photo to match the detected crop
   - darken:  tone-map the aligned result for projection

2. **Darken-only** (pre-cropped image):
   - darken:  tone-map the supplied image directly

Example usage:
    # Full pipeline
    python -m wzrd.prepare_surface day.jpg night.jpg -o surface.jpg

    # Darken-only
    python -m wzrd.prepare_surface cropped.jpg -o surface.jpg
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image

from .detect import detect_projection_area
from .align import align_images
from .darken import (
    darken_image,
    darken_image_file,
    MAX_BRIGHTNESS,
    DETAIL_BOOST,
    GUIDE_FRACTION,
    GUIDE_EPS,
    HIST_BINS,
    CHROMA_CORRECTION,
    GRADIENT_SOURCE,
    DEFAULT_ASPECT,
    BASE_RESOLUTION,
)


def prepare_surface(
    night_image_path: Union[str, Path],
    day_image_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    # detect params
    margin: float = 0.01,
    min_projection_fraction: float = 0.10,
    detect_aspect_ratio: Optional[str] = None,
    detect_resolution: Optional[int] = None,
    # darken params
    max_brightness: float = MAX_BRIGHTNESS,
    detail_boost: float = DETAIL_BOOST,
    guide_fraction: float = GUIDE_FRACTION,
    guide_eps: float = GUIDE_EPS,
    hist_bins: int = HIST_BINS,
    chroma_correction: float = CHROMA_CORRECTION,
    gradient_source: float = GRADIENT_SOURCE,
    target_aspect: str = DEFAULT_ASPECT,
    base_resolution: int = BASE_RESOLUTION,
    normalize: bool = True,
    alignment_aids: bool = True,
    # general
    verbose: bool = False,
    debug_dir: Optional[str] = None,
) -> Tuple[Image.Image, dict]:
    """
    Prepare a projection surface reference image.

    If *day_image_path* is provided, runs the full pipeline:
        detect(night) → align(day, crop) → darken(aligned)

    Otherwise runs darken-only on the supplied image (treated as
    an already-cropped surface photo).

    Args:
        night_image_path: In full-pipeline mode this is the nighttime photo
                          with the projector illuminating the surface.
                          In darken-only mode this is the pre-cropped image.
        day_image_path:   Optional daytime photo of the same surface.
                          When provided, the full pipeline is used.
        output_path:      Where to save the final darkened image.

        margin:                  Detect margin (fraction to shrink inward).
        min_projection_fraction: Minimum projection area fraction for detect.
        detect_aspect_ratio:     Target aspect ratio for detect crop (e.g. "16:9").
        detect_resolution:       Max dimension for detect output.

        max_brightness:    Darken luminance ceiling (0.0-1.0).
        detail_boost:      Detail layer amplification.
        guide_fraction:    Guided filter radius fraction.
        guide_eps:         Guided filter regularisation.
        hist_bins:         CDF histogram bins.
        chroma_correction: Chroma scaling strength.
        gradient_source:   Raw-L (0) vs detail-residual (1) gradient blend.
        target_aspect:     Aspect ratio for darken normalisation.
        base_resolution:   Max dimension for darken output.
        normalize:         Whether darken normalises aspect/resolution.
        alignment_aids:    Whether to generate alignment aid overlays.

        verbose:   Print progress messages.
        debug_dir: Save intermediate debug images to this directory.

    Returns:
        Tuple of (darkened_image_pil, info_dict)
    """
    info: dict = {'mode': 'darken_only'}

    if day_image_path is not None:
        # ── Full pipeline: detect → align → darken ──────────────────
        info['mode'] = 'full'

        # Step 1: Detect projection area in night image
        if verbose:
            print(f"[1/3] Detecting projection surface in {night_image_path} ...")

        cropped_bgr, detect_info = detect_projection_area(
            night_image_path,
            margin=margin,
            debug_dir=debug_dir,
            min_projection_fraction=min_projection_fraction,
            target_aspect_ratio=detect_aspect_ratio,
            output_resolution=detect_resolution,
        )
        info['detect'] = {k: v for k, v in detect_info.items() if k != 'preview'}

        if verbose:
            print(f"  Detected crop: {detect_info['cropped_size']}")

        # Step 2: Align day image to the detected crop
        if verbose:
            print(f"[2/3] Aligning {day_image_path} to detected crop ...")

        day_bgr = cv2.imread(str(day_image_path))
        if day_bgr is None:
            raise FileNotFoundError(f"Could not load day image: {day_image_path}")

        aligned_bgr, align_result = align_images(
            day_bgr, cropped_bgr, verbose=verbose,
        )
        info['align'] = {
            'method': align_result.method,
            'num_matches': align_result.num_matches,
            'num_inliers': align_result.num_inliers,
            'confidence': align_result.confidence,
        }

        if verbose:
            print(f"  Alignment: {align_result.method}, "
                  f"confidence={align_result.confidence:.1%}")

        # Step 3: Darken the aligned image
        if verbose:
            print(f"[3/3] Darkening aligned image ...")

        # align_images returns BGR; darken_image expects RGB or PIL
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        from .utils import DebugContext
        debug = DebugContext(debug_dir)

        # When we already have a precisely cropped + aligned array,
        # optionally normalise to the target aspect/resolution via PIL first.
        pil_img = Image.fromarray(aligned_rgb)
        if normalize:
            from .utils import parse_aspect_ratio, normalize_image
            aspect_val = parse_aspect_ratio(target_aspect)
            pil_img = normalize_image(pil_img, aspect_val, base_resolution)

        darkened_arr = darken_image(
            pil_img,
            max_brightness=max_brightness,
            detail_boost=detail_boost,
            guide_fraction=guide_fraction,
            guide_eps=guide_eps,
            hist_bins=hist_bins,
            chroma_correction=chroma_correction,
            gradient_source=gradient_source,
            debug=debug,
        )
        result = Image.fromarray(darkened_arr)

        if alignment_aids:
            from .utils import _generate_alignment_aids
            surface_rgb = np.array(pil_img)
            if output_path is not None:
                aids_output_dir = str(Path(output_path).parent)
                aids_stem = Path(output_path).stem
            else:
                aids_output_dir = str(Path(night_image_path).parent)
                aids_stem = Path(night_image_path).stem + "_surface"
            analysis = _generate_alignment_aids(
                surface_rgb, output_dir=aids_output_dir, stem=aids_stem,
            )
            if 'video_path' in analysis:
                info['alignment_video_path'] = analysis['video_path']

    else:
        # ── Darken-only mode ────────────────────────────────────────
        if verbose:
            print(f"[1/1] Darkening {night_image_path} ...")

        result, darken_info = darken_image_file(
            night_image_path,
            max_brightness=max_brightness,
            detail_boost=detail_boost,
            guide_fraction=guide_fraction,
            guide_eps=guide_eps,
            hist_bins=hist_bins,
            chroma_correction=chroma_correction,
            gradient_source=gradient_source,
            target_aspect=target_aspect,
            base_resolution=base_resolution,
            normalize=normalize,
            alignment_aids=alignment_aids,
            debug_dir=debug_dir,
        )
        info['darken'] = darken_info

    info['final_size'] = result.size

    if output_path is not None:
        result.save(str(output_path), quality=95)
        info['output_path'] = str(output_path)
        if verbose:
            print(f"Saved: {output_path}  ({result.size[0]}x{result.size[1]})")

    return result, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    """CLI entry point for ``python -m wzrd.prepare_surface``."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare a projection surface reference image.\n\n'
                    'Full pipeline (two images):  detect → align → darken\n'
                    'Darken-only (one image):     darken only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional: always need at least one image
    parser.add_argument(
        'image',
        help='Image to process. In darken-only mode this is the pre-cropped '
             'surface image. In full-pipeline mode this is the DAYTIME photo.',
    )
    parser.add_argument(
        'night_image', nargs='?', default=None,
        help='Nighttime photo with projector illuminating the surface. '
             'When provided, the full detect→align→darken pipeline is used.',
    )
    parser.add_argument(
        '-o', '--output',
        help='Output path (default: <input>_surface.jpg)',
    )

    # Detect options
    detect_group = parser.add_argument_group('detection (full pipeline only)')
    detect_group.add_argument(
        '--margin', type=float, default=0.01,
        help='Shrink detected crop inward by this fraction (default: 0.01)',
    )
    detect_group.add_argument(
        '--min-projection-fraction', type=float, default=0.10,
        help='Minimum projection area as fraction of image (default: 0.10)',
    )
    detect_group.add_argument(
        '--detect-aspect-ratio', type=str, default=None,
        help='Target aspect ratio for detection crop (e.g. "16:9")',
    )
    detect_group.add_argument(
        '--detect-resolution', type=int, default=None,
        help='Max dimension for detection output (pixels)',
    )

    # Darken options
    darken_group = parser.add_argument_group('darkening')
    darken_group.add_argument(
        '--max-brightness', type=float, default=MAX_BRIGHTNESS,
        help=f'Luminance budget ceiling (default: {MAX_BRIGHTNESS})',
    )
    darken_group.add_argument(
        '--detail-boost', type=float, default=DETAIL_BOOST,
        help=f'Detail amplification (default: {DETAIL_BOOST})',
    )
    darken_group.add_argument(
        '--guide-fraction', type=float, default=GUIDE_FRACTION,
        help=f'Guided filter radius fraction (default: {GUIDE_FRACTION})',
    )
    darken_group.add_argument(
        '--guide-eps', type=float, default=GUIDE_EPS,
        help=f'Guided filter regularisation (default: {GUIDE_EPS})',
    )
    darken_group.add_argument(
        '--hist-bins', type=int, default=HIST_BINS,
        help=f'CDF histogram bins (default: {HIST_BINS})',
    )
    darken_group.add_argument(
        '--chroma-correction', type=float, default=CHROMA_CORRECTION,
        help=f'Chroma scaling strength (default: {CHROMA_CORRECTION})',
    )
    darken_group.add_argument(
        '--gradient-source', type=float, default=GRADIENT_SOURCE,
        help=f'Raw-L vs detail gradient blend (default: {GRADIENT_SOURCE})',
    )
    darken_group.add_argument(
        '--aspect', type=str, default=DEFAULT_ASPECT,
        help=f'Target aspect ratio (default: {DEFAULT_ASPECT})',
    )
    darken_group.add_argument(
        '--base-resolution', type=int, default=BASE_RESOLUTION,
        help=f'Base resolution (default: {BASE_RESOLUTION})',
    )
    darken_group.add_argument(
        '--no-normalize', action='store_true',
        help='Skip aspect ratio / resolution normalisation',
    )
    darken_group.add_argument(
        '--no-alignment-aids', action='store_true',
        help='Skip generation of alignment aid overlays',
    )

    # General
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print progress messages')
    parser.add_argument(
        '--debug', nargs='?', const='debug_output', default=None,
        metavar='DIR',
        help='Save debug images to directory (default: debug_output)',
    )

    args = parser.parse_args()

    # Resolve mode: if night_image is given → full pipeline
    if args.night_image is not None:
        day_path = args.image
        night_path = args.night_image
    else:
        day_path = None
        night_path = args.image

    # Default output path
    if args.output is None:
        p = Path(args.image)
        args.output = str(p.with_name(f"{p.stem}_surface.jpg"))

    try:
        result, info = prepare_surface(
            night_image_path=night_path,
            day_image_path=day_path,
            output_path=args.output,
            margin=args.margin,
            min_projection_fraction=args.min_projection_fraction,
            detect_aspect_ratio=args.detect_aspect_ratio,
            detect_resolution=args.detect_resolution,
            max_brightness=args.max_brightness,
            detail_boost=args.detail_boost,
            guide_fraction=args.guide_fraction,
            guide_eps=args.guide_eps,
            hist_bins=args.hist_bins,
            chroma_correction=args.chroma_correction,
            gradient_source=args.gradient_source,
            target_aspect=args.aspect,
            base_resolution=args.base_resolution,
            normalize=not args.no_normalize,
            alignment_aids=not args.no_alignment_aids,
            verbose=args.verbose,
            debug_dir=args.debug,
        )

        print(f"Mode: {info['mode']}")
        print(f"Output: {args.output}  ({info['final_size'][0]}x{info['final_size'][1]})")

        if 'detect' in info:
            print(f"  Detected crop: {info['detect']['cropped_size']}")
        if 'align' in info:
            ai = info['align']
            print(f"  Alignment: {ai['method']}, confidence={ai['confidence']:.1%}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(_main())
