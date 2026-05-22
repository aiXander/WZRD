"""
Layer-pack authoring — the offline → runtime data contract.

Takes a set of semantic mask PNGs (typically the output of
:func:`wzrd.islands.extract_color_regions`, possibly hand-edited / SAM-refined)
and a small hand-authored tags file, and emits a self-contained directory
that the realtime render-core consumes:

    layerpack-<name>/
      pack.json             # §4.1 layer-pack schema, version 1
      surface.png           # darkened / aligned surface (for preview overlay)
      masks/
        000_<id>.png        # antialiased grayscale, projector-resolution
        001_<id>.png
        ...
      references/
        photo.jpg           # original capture (optional)
        canny.png           # optional alignment aid

Identity is *semantic*: every layer has a stable ``id`` (assigned by the
human / agent in ``tags.json``), and the runtime contract guarantees that
``id``, ``tags``, ``group``, and ``parent`` survive re-shoots and
re-segmentations.  Mask filenames are an implementation detail (D7).

Tags file schema (all keys optional except per-mask ``id``):

    {
      "projector_resolution": [1920, 1080],
      "source_capture": "path/to/photo.jpg",
      "canny": "path/to/canny.png",
      "background": true,             // generate `000_background.png`
                                       // as the complement of all masks
      "layers": {
        "region_mask_001_color_000afe.webp": {
          "id": "trunk",
          "label": "trunk",
          "tags": ["tree", "structure"],
          "parent": null,
          "z": 1
        },
        ...
      },
      "groups": [
        {"id": "leaves", "members": ["leaf_a", "leaf_b"]}
      ]
    }

If a mask file isn't named in ``layers``, its ``id`` defaults to its stem
and it gets z-order = file order.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

LAYERPACK_VERSION = 1
MAX_LAYERS = 256  # D4: Texture2DArray hard cap


# ──────────────────────────────────────────────────────────────────────────────
# Mask discovery
# ──────────────────────────────────────────────────────────────────────────────

_MASK_EXTS = {".png", ".webp", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _discover_masks(masks_dir: Path) -> List[Path]:
    """Return mask files in deterministic order.

    If an ``islands.json`` sits in ``masks_dir`` (the wzrd.islands output
    convention), respect its ``regions`` ordering — that's the order the
    upstream segmenter chose.  Otherwise sort by filename.
    """
    islands_meta = masks_dir / "islands.json"
    if islands_meta.is_file():
        with open(islands_meta) as f:
            meta = json.load(f)
        ordered: List[Path] = []
        for region in meta.get("regions", []):
            name = region.get("region_mask")
            if not name:
                continue
            p = masks_dir / name
            if p.is_file():
                ordered.append(p)
        if ordered:
            return ordered

    return sorted(
        p for p in masks_dir.iterdir()
        if p.suffix.lower() in _MASK_EXTS
        and not p.name.startswith("quantized")
        and not p.name.startswith("region_crop_")  # secondary aid output
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mask normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read mask {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA — use alpha if present and non-trivial
            alpha = img[..., 3]
            if alpha.min() != alpha.max():
                img = alpha
            else:
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _normalise_mask(
    mask: np.ndarray,
    target_w: int,
    target_h: int,
    feather_px: int = 0,
) -> np.ndarray:
    """Resize to projector resolution, optionally feather edges.

    Result is uint8 grayscale.  The masks are *soft-edged* by design — the
    renderer wants antialiased boundaries to avoid pixel-grid seams between
    adjacent semantic regions.
    """
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(
            mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    if feather_px > 0:
        ksize = max(3, feather_px * 2 + 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_px / 2.0)

    return mask.astype(np.uint8)


def _bbox_and_centroid(mask: np.ndarray) -> Tuple[List[int], List[float], int]:
    """Return ([x0,y0,x1,y1], [cx,cy], area_px) for a grayscale mask.

    Bbox is computed on a thresholded binary mask (>=128).  Centroid is the
    *weighted* centroid (pixel value as weight), which is more stable for soft
    masks than the binary centroid.
    """
    binary = mask >= 128
    area_px = int(binary.sum())
    if area_px == 0:
        h, w = mask.shape[:2]
        return [0, 0, w, h], [w / 2.0, h / 2.0], 0

    ys, xs = np.where(binary)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    weights = mask.astype(np.float64)
    total = weights.sum()
    yy, xx = np.indices(mask.shape, dtype=np.float64)
    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)
    return [x0, y0, x1, y1], [cx, cy], area_px


# ──────────────────────────────────────────────────────────────────────────────
# Tags / identity resolution
# ──────────────────────────────────────────────────────────────────────────────

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _sanitize_id(raw: str) -> str:
    """Coerce a string into a safe layer id ([a-zA-Z0-9_-])."""
    cleaned = _SAFE_ID_RE.sub("_", raw).strip("_")
    return cleaned or "layer"


def _resolve_layer_meta(
    mask_path: Path,
    z_default: int,
    tags_layers: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Look up per-mask metadata from the tags file, with sensible defaults.

    Lookup keys tried in order: full filename, stem, prefix-stripped stem.
    """
    candidates = [
        mask_path.name,
        mask_path.stem,
    ]
    # `region_mask_001_color_xxxxxx` → `001_color_xxxxxx`
    if mask_path.stem.startswith("region_mask_"):
        candidates.append(mask_path.stem[len("region_mask_"):])

    entry: Dict[str, Any] = {}
    for k in candidates:
        if k in tags_layers:
            entry = dict(tags_layers[k])
            break

    if "id" not in entry:
        entry["id"] = _sanitize_id(mask_path.stem)
    else:
        entry["id"] = _sanitize_id(str(entry["id"]))
    entry.setdefault("label", entry["id"])
    entry.setdefault("tags", [])
    entry.setdefault("parent", None)
    entry.setdefault("z", z_default)
    return entry


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_layerpack(
    masks_dir: PathLike,
    output_dir: PathLike,
    surface: Optional[PathLike] = None,
    tags: Optional[Union[PathLike, Dict[str, Any]]] = None,
    projector_resolution: Optional[Tuple[int, int]] = None,
    source_capture: Optional[PathLike] = None,
    canny: Optional[PathLike] = None,
    feather_px: int = 1,
    include_background: bool = False,
) -> Dict[str, Any]:
    """Emit a layer pack from masks + surface + tags.

    Args:
        masks_dir: Directory containing mask images (raw, or a
            ``wzrd.islands.extract_color_regions`` output dir).
        output_dir: Destination — created if absent, files written into it.
        surface: Path to the darkened / aligned surface image (becomes
            ``surface.png`` in the pack).  Optional but recommended; if
            absent the projector_resolution must be set explicitly.
        tags: Path to a tags JSON file *or* an already-parsed dict.  See the
            module docstring for the schema.
        projector_resolution: Override (width, height).  Defaults to the
            surface image resolution, or the first mask's resolution if
            no surface is supplied.
        source_capture: Path to the original photo (copied into
            ``references/photo<ext>`` for preview overlays).
        canny: Path to a canny/edge reference (copied into ``references/``).
        feather_px: Gaussian-blur radius applied to each mask after resize.
            Set to 0 to keep hard edges.
        include_background: If true, generates ``000_background.png`` as the
            complement of the union of all input masks.

    Returns:
        The serialised ``pack.json`` manifest content as a dict.
    """
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks_dir not found: {masks_dir}")

    if isinstance(tags, (str, Path)):
        with open(tags) as f:
            tags_data: Dict[str, Any] = json.load(f)
    elif isinstance(tags, dict):
        tags_data = tags
    else:
        tags_data = {}

    tags_layers: Dict[str, Dict[str, Any]] = tags_data.get("layers", {}) or {}
    tags_groups: List[Dict[str, Any]] = tags_data.get("groups", []) or []

    if projector_resolution is None:
        if "projector_resolution" in tags_data:
            projector_resolution = tuple(tags_data["projector_resolution"])  # type: ignore[assignment]
        elif surface is not None:
            surf_arr = cv2.imread(str(surface), cv2.IMREAD_UNCHANGED)
            if surf_arr is None:
                raise FileNotFoundError(f"could not read surface {surface}")
            projector_resolution = (surf_arr.shape[1], surf_arr.shape[0])
        else:
            # Last resort: use the first mask's dimensions
            mask_files = _discover_masks(masks_dir)
            if not mask_files:
                raise ValueError(f"no masks found in {masks_dir}")
            first = _load_grayscale(mask_files[0])
            projector_resolution = (first.shape[1], first.shape[0])
    pw, ph = int(projector_resolution[0]), int(projector_resolution[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_out = output_dir / "masks"
    masks_out.mkdir(exist_ok=True)
    refs_out = output_dir / "references"

    # Surface
    surface_rel: Optional[str] = None
    if surface is not None:
        surf_arr = cv2.imread(str(surface), cv2.IMREAD_COLOR)
        if surf_arr is None:
            raise FileNotFoundError(f"could not read surface {surface}")
        if (surf_arr.shape[1], surf_arr.shape[0]) != (pw, ph):
            surf_arr = cv2.resize(surf_arr, (pw, ph), interpolation=cv2.INTER_AREA)
        surface_rel = "surface.png"
        cv2.imwrite(str(output_dir / surface_rel), surf_arr)

    # Reference images
    ref_capture_rel: Optional[str] = None
    if source_capture is None and tags_data.get("source_capture"):
        source_capture = tags_data["source_capture"]
    if source_capture is not None:
        src = Path(source_capture)
        if src.is_file():
            refs_out.mkdir(exist_ok=True)
            ref_capture_rel = f"references/photo{src.suffix.lower()}"
            shutil.copy2(src, output_dir / ref_capture_rel)

    ref_canny_rel: Optional[str] = None
    if canny is None and tags_data.get("canny"):
        canny = tags_data["canny"]
    if canny is not None:
        cny = Path(canny)
        if cny.is_file():
            refs_out.mkdir(exist_ok=True)
            ref_canny_rel = f"references/canny{cny.suffix.lower()}"
            shutil.copy2(cny, output_dir / ref_canny_rel)

    # Masks
    mask_files = _discover_masks(masks_dir)
    if not mask_files:
        raise ValueError(f"no mask files found in {masks_dir}")

    include_bg = include_background or bool(tags_data.get("background", False))
    layers: List[Dict[str, Any]] = []
    seen_ids: Dict[str, Path] = {}
    union = np.zeros((ph, pw), dtype=np.uint16) if include_bg else None

    slice_index = 1 if include_bg else 0
    for mask_path in mask_files:
        if slice_index >= MAX_LAYERS:
            raise ValueError(
                f"layer pack exceeds the {MAX_LAYERS}-slice Texture2DArray cap (D4)"
            )

        raw = _load_grayscale(mask_path)
        mask = _normalise_mask(raw, pw, ph, feather_px=feather_px)
        bbox, centroid, area_px = _bbox_and_centroid(mask)

        meta = _resolve_layer_meta(mask_path, z_default=slice_index, tags_layers=tags_layers)
        layer_id = meta["id"]
        if layer_id in seen_ids:
            raise ValueError(
                f"duplicate layer id {layer_id!r} "
                f"(mask {mask_path.name} collides with {seen_ids[layer_id].name})"
            )
        seen_ids[layer_id] = mask_path

        out_name = f"{slice_index:03d}_{layer_id}.png"
        out_path = masks_out / out_name
        cv2.imwrite(str(out_path), mask)

        if union is not None:
            union += mask  # uint16 saturation-safe accumulation

        layers.append({
            "id": layer_id,
            "mask": f"masks/{out_name}",
            "label": meta["label"],
            "tags": list(meta["tags"]),
            "bbox": bbox,
            "centroid": centroid,
            "area_px": area_px,
            "parent": meta["parent"],
            "z": int(meta["z"]),
        })
        slice_index += 1

    if include_bg and union is not None:
        bg_mask = np.clip(255 - np.minimum(union, 255).astype(np.int16), 0, 255).astype(np.uint8)
        if feather_px > 0:
            ksize = max(3, feather_px * 2 + 1)
            bg_mask = cv2.GaussianBlur(bg_mask, (ksize, ksize), feather_px / 2.0)
        bbox, centroid, area_px = _bbox_and_centroid(bg_mask)
        bg_name = "000_background.png"
        cv2.imwrite(str(masks_out / bg_name), bg_mask)
        layers.insert(0, {
            "id": "background",
            "mask": f"masks/{bg_name}",
            "label": "background",
            "tags": ["background"],
            "bbox": bbox,
            "centroid": centroid,
            "area_px": area_px,
            "parent": None,
            "z": 0,
        })

    # Groups — only emit those whose members all resolve to known layer ids
    known = set(seen_ids.keys()) | ({"background"} if include_bg else set())
    groups: List[Dict[str, Any]] = []
    for g in tags_groups:
        gid = g.get("id")
        members = [m for m in g.get("members", []) if m in known]
        if gid and members:
            groups.append({"id": _sanitize_id(str(gid)), "members": members})

    scene: Dict[str, Any] = {
        "version": LAYERPACK_VERSION,
        "projector_resolution": [pw, ph],
        "source_capture": ref_capture_rel,
        "surface": surface_rel,
        "layers": layers,
        "groups": groups,
    }
    if ref_canny_rel:
        scene["references"] = {"canny": ref_canny_rel}

    with open(output_dir / "pack.json", "w") as f:
        json.dump(scene, f, indent=2)

    return scene


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        prog="python -m wzrd.layerpack",
        description="Build a WZRD layer pack from masks + surface + tags.",
    )
    p.add_argument("masks_dir", help="directory of mask PNGs (or wzrd.islands output)")
    p.add_argument("-o", "--output", required=True, help="output pack directory")
    p.add_argument("--surface", help="path to darkened/aligned surface image")
    p.add_argument("--tags", help="tags JSON mapping mask filenames → semantic info")
    p.add_argument("--projector-resolution", help="WxH override, e.g. 1920x1080")
    p.add_argument("--source-capture", help="original photo (copied into references/)")
    p.add_argument("--canny", help="optional canny/edge aid (copied into references/)")
    p.add_argument("--feather", type=int, default=1, help="mask feather radius in px (default: 1)")
    p.add_argument("--background", action="store_true",
                   help="auto-generate 000_background.png as the complement of all masks")
    args = p.parse_args()

    projector_resolution = None
    if args.projector_resolution:
        try:
            w_s, h_s = args.projector_resolution.lower().split("x")
            projector_resolution = (int(w_s), int(h_s))
        except ValueError:
            p.error("--projector-resolution must look like 1920x1080")

    scene = build_layerpack(
        masks_dir=args.masks_dir,
        output_dir=args.output,
        surface=args.surface,
        tags=args.tags,
        projector_resolution=projector_resolution,
        source_capture=args.source_capture,
        canny=args.canny,
        feather_px=args.feather,
        include_background=args.background,
    )
    n_layers = len(scene["layers"])
    pw, ph = scene["projector_resolution"]
    print(f"wrote layer pack to {args.output}: {n_layers} layers @ {pw}x{ph}")


if __name__ == "__main__":
    _cli()
