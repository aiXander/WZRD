# External resources

A running index of external projects, papers and tools that may be useful for WZRD —
surface extraction, depth/geometry estimation, generative video, projection mapping, realtime GPU work.

**This is a link parking lot, not a dependency list.** Nothing here is used by the codebase
unless a reference doc says so. Append new entries at the top of the relevant section with a
one-or-two-sentence "why we'd care" note — the description should say what it buys *us*, not
just what the project is.

## Depth / geometry / surface understanding

### Marigold V2 — <https://github.com/huawei-bayerlab/marigold-v2>

Diffusion-transformer family for dense prediction from a single image: affine-invariant depth
(log/linear/disparity), surface normals, albedo, "see-through" depth (geometry behind glass), and
metric depth completion. Built as LoRA adapters on a frozen Qwen-Image-Edit-2509 backbone, so
fine-tuning to a new dense task is claimed to take <1 week on one consumer GPU. Apache-2.0 (code +
adapters; backbone and datasets keep their own licenses).

*Why we'd care:* the offline toolkit currently reasons about a projection surface in 2D only
(`wzrd/detect.py` colour+Otsu+hull, `wzrd/align.py` homography). Per-pixel depth/normals/albedo of
a real surface photo would give a geometric prior for warping and relighting content to the actual
wall — and the albedo/normal outputs are the missing half of "make projected light sit correctly on
a textured physical surface". Also plausible as a smarter mask/island source than KMeans
(`wzrd/islands.py`).
