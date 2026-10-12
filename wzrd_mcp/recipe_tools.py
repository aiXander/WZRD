"""Style recipes — the global TextureFlow look library (gig P7, contract C6).

A recipe is one folder under `library/recipes/` (`WZRD_RECIPES_ROOT`
overrides the root; the default is what a Finder-launched app sees):

    library/recipes/<name>/
      recipe.json   {"schema", "name", "when_to_use", "notes", "prompt",
                     "images": [file names in this folder], "args": {...},
                     "source": {...} | null}
      style_1.jpg … the style images, downscaled (they steer IP-Adapter, which
                    sees 224 px — full-size copies would only bloat git)
      preview.jpg   one frame of a render with it

`args` are the look — TextureFlow arguments minus everything that belongs to
the target rather than the style: media (`images` lives beside it,
`control_input` / `diffusion_mask` come from the region), `width`/`height`
(the region's aspect) and `seed` (kept in `source` for reproduction).
`prompt` is the image prompt the style images were made with, when known,
so an agent can make more in the same vein.

Hand-curated and agent-saved recipes share the folder: `save_recipe` builds
one from a clip's provenance sidecar (C2), `texture_flow(recipe=…)` takes its
args as defaults. Large media never goes in (no preview clips) — the
`library/recipes/.gitignore` keeps video out of git.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from . import project
from .provenance import THUMB_SUFFIX, sidecar_path
from .server import tool

RECIPE_FILE = "recipe.json"
RECIPE_SCHEMA = 1
PREVIEW = "preview.jpg"
STYLE_MAX_SIDE = 768  # IP-Adapter's CLIP sees 224 px; 768 keeps detail for a human eye
PREVIEW_WIDTH = 640

#: Not part of a look: media, the target's frame, the dice, and WZRD-local args.
_TARGET_ARGS = {"images", "control_input", "diffusion_mask", "width", "height", "seed",
                "region", "allow_over_cap", "hap", "recipe", "model"}


def recipes_root() -> Path:
    env = os.getenv("WZRD_RECIPES_ROOT")
    root = Path(env).expanduser() if env else project._REPO_ROOT / "library" / "recipes"
    return root.resolve()


def _recipe_dir(name: str) -> Path:
    return recipes_root() / project.slugify(name).lower()


def load_recipe(name: str) -> dict:
    """recipe.json of *name*, with `images` resolved to absolute paths."""
    path = _recipe_dir(name) / RECIPE_FILE
    if not path.is_file():
        known = ", ".join(r["name"] for r in _all_recipes()) or "none yet"
        raise ToolError(f"no recipe {name!r} in {recipes_root()} (recipes: {known})")
    recipe = json.loads(path.read_text())
    recipe["images"] = [i if i.startswith(("http://", "https://")) else str(path.parent / i)
                        for i in recipe.get("images", [])]
    missing = [i for i in recipe["images"] if i.startswith("/") and not Path(i).is_file()]
    if missing:
        raise ToolError(f"recipe {name!r} lists style images that aren't there: {', '.join(missing)}")
    return recipe


def apply_recipe(name: str, args: dict) -> dict:
    """TextureFlow args for a call with `recipe=name`: the recipe's args and
    style images as defaults, the caller's args (already non-null) win."""
    recipe = load_recipe(name)
    merged = {**recipe.get("args", {}), **args}
    merged.setdefault("images", recipe["images"])
    if not merged.get("images"):
        raise ToolError(f"recipe {name!r} has no style images — pass images=")
    shaped = merged.get("use_controlnet1") or float(merged.get("denoise", 1.0)) < 1.0
    if shaped and not merged.get("control_input"):
        raise ToolError(
            f"recipe {name!r} shapes the render from a control_input (use_controlnet1 / denoise < 1) — "
            f"pass control_input (the region's region_surface_image, or the dark surface)"
        )
    return merged


def _all_recipes() -> list[dict]:
    root = recipes_root()
    out = []
    for path in sorted(root.glob(f"*/{RECIPE_FILE}")):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        data["_dir"] = path.parent
        out.append(data)
    return out


def _look(args: dict) -> str:
    """One-line summary of a recipe's args for the table."""
    model = Path(str(args.get("base_model", ""))).stem.split("_")[0] or "default model"
    bits = [model]
    if "mapping_mode" in args:
        bits.append(args["mapping_mode"])
    if "motion_scale" in args:
        bits.append(f"motion {args['motion_scale']}")
    if args.get("use_controlnet1"):
        bits.append(f"controlnet {args.get('preprocessor1', 'scribble')} {args.get('controlnet_strength1', '')}".strip())
    if float(args.get("denoise", 1.0)) < 1.0:
        bits.append(f"denoise {args['denoise']}")
    if args.get("use_upscale") is False and args.get("upscale_esrgan") is False:
        bits.append("no upscale")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# Sidecar walk: from a clip (.mov / .mp4 / thumb) back to the render's call
# ---------------------------------------------------------------------------
def _render_sidecar(clip: Path) -> tuple[Path, dict]:
    """(rendered video, its texture_flow sidecar), following `parent` links
    from *clip* (a HAP .mov made by to_hap points at its mp4)."""
    if clip.name.endswith(THUMB_SUFFIX):
        stem = clip.name[: -len(THUMB_SUFFIX)]
        candidates = [clip.with_name(stem + ext) for ext in (".mp4", ".mov")]
        clip = next((c for c in candidates if c.is_file()), candidates[0])
    path, seen = clip.resolve(), set()
    while path and str(path) not in seen:
        seen.add(str(path))
        meta_path = sidecar_path(path)
        if not meta_path.is_file():
            break
        meta = json.loads(meta_path.read_text())
        if meta.get("tool") == "texture_flow":
            return path, meta
        path = Path(meta["parent"]) if meta.get("parent") else None
    raise ToolError(
        f"{clip} has no texture_flow provenance (no <file>.meta.json chain back to a texture_flow "
        f"call) — a recipe is built from what the render was called with"
    )


def _style_copy(src: str, dst: Path) -> None:
    from PIL import Image

    from wzrd.utils import load_upright

    img = load_upright(src).convert("RGB")
    img.thumbnail((STYLE_MAX_SIDE, STYLE_MAX_SIDE), Image.LANCZOS)
    img.save(dst, quality=90)


def _image_prompt(image: str) -> Optional[str]:
    """The prompt a style image was generated with, from its own sidecar."""
    try:
        meta = json.loads(sidecar_path(Path(image).resolve()).read_text())
    except (OSError, ValueError):
        return None
    return (meta.get("args") or {}).get("prompt")


def _save(name: str, clip: Path, when_to_use: str, notes: Optional[str], prompt: Optional[str],
          overwrite: bool) -> dict:
    from wzrd.clips import thumbnail

    video, meta = _render_sidecar(clip)
    args = meta.get("args") or {}
    images = args.get("images") or []
    if not images:
        raise ToolError(f"{video.name}'s sidecar records no style images")
    local = [i for i in images if not i.startswith(("http://", "https://"))]
    missing = [i for i in local if not Path(i).expanduser().is_file()]
    if missing:
        raise ToolError(f"style images of {video.name} are gone: {', '.join(missing)}")

    dest = _recipe_dir(name)
    if dest.exists() and not overwrite:
        raise ToolError(f"recipe {dest.name!r} exists — pick another name, or overwrite=true to replace it")
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        names = []
        for i, ref in enumerate(images, 1):
            if ref.startswith(("http://", "https://")):
                names.append(ref)  # TextureFlow downloads URLs itself
                continue
            fname = f"style_{i}.jpg"
            _style_copy(str(Path(ref).expanduser()), tmp / fname)
            names.append(fname)
        thumbnail(video, tmp / PREVIEW, width=PREVIEW_WIDTH)
        prompts = [p for p in (_image_prompt(i) for i in local) if p]
        recipe = {
            "schema": RECIPE_SCHEMA,
            "name": dest.name,
            "when_to_use": when_to_use,
            "notes": notes,
            "prompt": prompt or (prompts[0] if prompts else None),
            "images": names,
            "args": {k: v for k, v in args.items() if k not in _TARGET_ARGS},
            "source": {
                "clip": str(video),
                "project": video.parent.parent.name,
                "seed": args.get("seed"),
                "size": [args.get("width"), args.get("height")],
                "region": meta.get("region"),
                "rendered": meta.get("created"),
                "saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }
        (tmp / RECIPE_FILE).write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"recipe": dest.name, "dir": str(dest), "args": recipe["args"],
            "images": [str(dest / n) if not n.startswith("http") else n for n in names],
            "preview": str(dest / PREVIEW), "prompt": recipe["prompt"]}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool(title="List style recipes", tags={"generate", "textureflow", "library"}, read_only=True, idempotent=True)
async def list_recipes(name: Optional[str] = None, ctx: Optional[Context] = None) -> dict:
    """The style-recipe library: saved TextureFlow looks (style images + settings) to render new clips from with texture_flow(recipe=…).

    Without `name`: one compact row per recipe — name, when to use it, the
    look in a few words, image count. With `name`: that recipe in full
    (args, absolute style-image paths, notes, the image prompt, where it came
    from) — `view_media` its preview before choosing.

    Args:
        name: A recipe to show in full. Omit for the table.
    """
    if name:
        recipe = load_recipe(name)
        return {**recipe, "dir": str(_recipe_dir(name)),
                "preview": str(_recipe_dir(name) / PREVIEW) if (_recipe_dir(name) / PREVIEW).is_file() else None}
    rows = [{
        "name": r["name"],
        "when_to_use": r.get("when_to_use"),
        "look": _look(r.get("args") or {}),
        "images": len(r.get("images") or []),
    } for r in _all_recipes()]
    return {"root": str(recipes_root()), "recipes": rows}


@tool(title="Save style recipe", tags={"generate", "textureflow", "library"})
async def save_recipe(
    name: str,
    from_clip: str,
    when_to_use: str,
    notes: Optional[str] = None,
    prompt: Optional[str] = None,
    overwrite: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Save the look of a TextureFlow clip the operator liked as a reusable recipe in the global library (library/recipes/<name>/).

    Reads the clip's provenance sidecar (follows a HAP .mov back to its mp4),
    copies its style images (downscaled) and a preview frame, and keeps the
    style args — not the size, seed, region or masks, which belong to the
    target. Only save on the operator's word; recipes are shared across every
    project.

    Args:
        name: Recipe name (lowercase-dashes, e.g. "ember-veil").
        from_clip: Path of the clip — its .mov, .mp4 or .thumb.jpg in a project's generated/.
        when_to_use: One line a future agent picks it by: mood, motion, which surfaces/regions it reads on.
        notes: Optional craft notes: what worked, what to avoid, settings worth varying.
        prompt: The image prompt behind the style images; omit to take it from their own sidecars.
        overwrite: Replace an existing recipe of that name.
    """
    clip = Path(from_clip).expanduser()
    if not clip.is_file():
        raise ToolError(f"from_clip {from_clip!r} is not a file")
    return await asyncio.to_thread(_save, name, clip, when_to_use.strip(), notes, prompt, overwrite)

