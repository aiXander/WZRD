#!/usr/bin/env python3
"""Smoke-test script for WZRD submodules."""

import argparse
import os
import shutil
import sys
import time
import traceback

# ── Dummy input paths (update these to match your test assets) ──────────────
TEST_IMAGE = "test_imgs/day.jpg"
TEST_GENERATED = "test_imgs/generated.jpg"
TEST_VIDEO = "test_imgs/test_video/video.mp4"
TEST_BACKGROUND = "test_imgs/test_video/background.jpg"
TEST_SURFACE_PHOTO = "test_imgs/input_raw.png"
TEST_TARGET_CROP = "test_imgs/test_video/target_crop.jpg"

RESULTS_ROOT = "test_results"


# ── Per-module test functions ───────────────────────────────────────────────

def test_darken():
    from wzrd import darken
    _require(TEST_IMAGE)
    out_dir = _result_dir("darken")
    _copy_input(TEST_IMAGE, out_dir)
    img, info = darken.darken_image_file(TEST_IMAGE, output_path=os.path.join(out_dir, "output.jpg"))
    assert img is not None, "darken returned None image"


def test_subtract_frame():
    from wzrd import subtract_frame
    _require(TEST_GENERATED, TEST_BACKGROUND)
    out_dir = _result_dir("subtract_frame")
    _copy_input(TEST_GENERATED, out_dir)
    _copy_input(TEST_BACKGROUND, out_dir)
    img, info = subtract_frame.subtract_background_file(
        TEST_GENERATED, TEST_BACKGROUND,
        output_path=os.path.join(out_dir, "output.png"),
    )
    assert img is not None, "subtract_frame returned None image"


def test_subtract_video():
    from wzrd import subtract_video
    _require(TEST_VIDEO, TEST_BACKGROUND)
    out_dir = _result_dir("subtract_video")
    _copy_input(TEST_VIDEO, out_dir)
    _copy_input(TEST_BACKGROUND, out_dir)
    out = os.path.join(out_dir, "output.mp4")
    info = subtract_video.subtract_background_video(TEST_VIDEO, TEST_BACKGROUND, output_path=out)
    assert os.path.isfile(out), f"output video not created: {out}"


def test_islands():
    from wzrd import islands
    _require(TEST_IMAGE)
    out_dir = _result_dir("islands")
    _copy_input(TEST_IMAGE, out_dir)
    regions_dir = os.path.join(out_dir, "regions")
    regions, _ = islands.extract_color_regions(TEST_IMAGE, regions_dir)
    assert isinstance(regions, list), "islands did not return a list"


def test_reproject():
    from wzrd import reproject
    _require(TEST_VIDEO)
    out_dir = _result_dir("reproject")
    _copy_input(TEST_VIDEO, out_dir)
    out = os.path.join(out_dir, "output.mp4")
    metadata = {"x": 0, "y": 0, "width": 320, "height": 240}
    info = reproject.reproject_video(TEST_VIDEO, metadata, output_path=out)
    assert os.path.isfile(out), f"output video not created: {out}"


def test_detect():
    from wzrd import detect
    _require(TEST_SURFACE_PHOTO)
    out_dir = _result_dir("detect")
    _copy_input(TEST_SURFACE_PHOTO, out_dir)
    cropped, info = detect.detect_projection_area(
        TEST_SURFACE_PHOTO,
        output_path=os.path.join(out_dir, "output.jpg"),
    )
    assert cropped is not None, "detect returned None image"


def test_align():
    from wzrd import align
    _require(TEST_SURFACE_PHOTO, TEST_TARGET_CROP)
    out_dir = _result_dir("align")
    _copy_input(TEST_SURFACE_PHOTO, out_dir)
    _copy_input(TEST_TARGET_CROP, out_dir)
    warped, info = align.align_images_file(
        TEST_SURFACE_PHOTO, TEST_TARGET_CROP,
        output_path=os.path.join(out_dir, "output.jpg"),
    )
    assert warped is not None, "align returned None image"


# ── Test registry ───────────────────────────────────────────────────────────

TESTS = {
    "darken": test_darken,
    "subtract_frame": test_subtract_frame,
    "subtract_video": test_subtract_video,
    "islands": test_islands,
    "reproject": test_reproject,
    "detect": test_detect,
    "align": test_align,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

class _MissingFile(Exception):
    pass


def _require(*paths):
    for p in paths:
        if not os.path.isfile(p):
            raise _MissingFile(f"required input file not found: {p}")


def _result_dir(test_name):
    """Create and return test_results/<test_name>/, clearing any previous run."""
    d = os.path.join(RESULTS_ROOT, test_name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    return d


def _copy_input(src, dst_dir):
    """Copy an input file into dst_dir, preserving the basename."""
    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def _run_test(name, fn):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  PASS  ({elapsed:.2f}s)")
        return True
    except _MissingFile as e:
        elapsed = time.perf_counter() - t0
        print(f"  SKIP  {e}  ({elapsed:.2f}s)")
        return None  # skip
    except Exception:
        elapsed = time.perf_counter() - t0
        traceback.print_exc()
        print(f"  FAIL  ({elapsed:.2f}s)")
        return False


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WZRD smoke tests")
    parser.add_argument("modules", nargs="*", metavar="MODULE",
                        help=f"modules to test: {', '.join(TESTS)}")
    parser.add_argument("--all", action="store_true", help="test every module")
    args = parser.parse_args()

    if args.all:
        selected = list(TESTS)
    elif args.modules:
        bad = [m for m in args.modules if m not in TESTS]
        if bad:
            parser.error(f"unknown module(s): {', '.join(bad)}")
        selected = args.modules
    else:
        parser.print_help()
        sys.exit(0)

    passed, failed, skipped = 0, 0, 0
    for name in selected:
        result = _run_test(name, TESTS[name])
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
