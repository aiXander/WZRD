"""Probe Nano Banana 2.1 called directly on Google (Gemini API + Vertex), 2026-10-09.

Evidence behind docs/reference/hosted-models.md (the Google move): thinking
levels and their draft images, multi-ref labelling, seed, search grounding,
reference-count and payload limits, 4K timing, Vertex `thought` flags. Reuses
the FAL shootout's prompts/refs so outputs sit next to FAL's in one page.

    uv run --no-project --with google-genai --with pillow --with numpy \
        python scripts/google_nb_probe.py

Writes projects/google-nb-tests/{<test>/..., results.json, index.html}.
Every image part is saved, drafts included (suffix `_draft`). ~13 calls, ~$0.70.
"""
from __future__ import annotations

import asyncio
import html
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "projects" / "google-nb-tests"
SHOOTOUT = REPO / "projects" / "fal-shootout"
STORY = REPO / "test_imgs" / "stories" / "story_01"
CHARACTER = STORY / "character.jpg"
BUILDING = STORY / "building.jpg"
KASBAH = REPO / "test_imgs" / "morocco" / "kasbah" / "input_raw_darkened.jpg"
CREATURES = sorted((REPO / "test_imgs" / "creatures").glob("*.jpg"))
MODEL = "gemini-nano-banana-2.1"
VERTEX_PROJECT = "gen-lang-client-0837456283"

STYLE_LOCK = ("Keep the creature's exact design, proportions, glow and hand-drawn "
              "colored-pencil style.")
A5_PROMPT = ("Add the glowing creature from image_2 onto the large wall of the "
             "building in image_1, as if it were light projected onto the wall. "
             "Keep every other part of image_1 exactly unchanged: same framing, "
             "same darkness, same wall texture and sky. Only the creature is added. "
             f"{STYLE_LOCK}")
A1_PROMPT = ("Place the glowing creature from image_1 in front of the building in "
             "image_2, standing on the ground before the central doorway. "
             f"{STYLE_LOCK} Match the night lighting and color palette of image_2, "
             "and let the creature's glow cast soft light onto the stone facade.")


def _env() -> dict:
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


ENV = _env()
GEMINI = genai.Client(api_key=ENV["GEMINI_API_KEY"])
VERTEX = genai.Client(vertexai=True, api_key=ENV["VERTEX_API_KEY"],
                      project=VERTEX_PROJECT, location="global")
SEM = asyncio.Semaphore(4)


def _mime(p: Path) -> str:
    return "image/png" if p.suffix.lower() == ".png" else "image/jpeg"


def contents(prompt: str, refs: list[Path | bytes]) -> list[types.Part]:
    parts: list[types.Part] = []
    for i, r in enumerate(refs, 1):
        data, mime = (r, "image/png") if isinstance(r, bytes) else (r.read_bytes(), _mime(r))
        parts += [types.Part(text=f"image_{i}:"),
                  types.Part(inline_data=types.Blob(mime_type=mime, data=data))]
    return parts + [types.Part(text=prompt)]


def surface_stability(img: Path) -> float:
    """Share of pixels changed vs the darkened kasbah (>25/255) — same metric as
    scripts/fal_shootout.py. Lower = surface left alone."""
    out = Image.open(img).convert("RGB")
    ref = Image.open(KASBAH).convert("RGB").resize(out.size, Image.LANCZOS)
    diff = np.abs(np.asarray(out, np.int16) - np.asarray(ref, np.int16)).max(axis=2)
    return round(float((diff > 25).mean()), 3)


async def call(test: str, tag: str, prompt: str, refs: list, *, client=GEMINI,
               aspect: str | None = None, size: str = "1K", thinking: str | None = None,
               seed: int | None = None, search: bool = False, note: str = "") -> dict:
    cfg = dict(response_modalities=["TEXT", "IMAGE"],
               image_config=types.ImageConfig(aspect_ratio=aspect, image_size=size))
    if thinking:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    if seed is not None:
        cfg["seed"] = seed
    if search:
        cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    rec = {"test": test, "tag": tag, "backend": "vertex" if client is VERTEX else "gemini",
           "aspect": aspect, "size": size, "thinking": thinking, "seed": seed,
           "search": search, "n_refs": len(refs), "note": note, "images": [], "texts": []}
    d = OUT / test
    d.mkdir(parents=True, exist_ok=True)
    async with SEM:
        t0 = time.time()
        try:
            resp = await client.aio.models.generate_content(
                model=MODEL, contents=contents(prompt, refs),
                config=types.GenerateContentConfig(**cfg))
        except Exception as e:  # noqa: BLE001 — a probe records failures
            rec.update(wall_s=round(time.time() - t0, 1), error=f"{type(e).__name__}: {str(e)[:400]}")
            print(f"[{test}/{tag}] ERROR {rec['error'][:160]}")
            return rec
        rec["wall_s"] = round(time.time() - t0, 1)
    cand = (resp.candidates or [None])[0]
    rec["finish_reason"] = str(getattr(cand, "finish_reason", None))
    parts = (cand.content.parts if cand and cand.content else None) or []
    img_idx = [i for i, p in enumerate(parts) if p.inline_data]
    for i, p in enumerate(parts):
        if p.inline_data:
            final = i == img_idx[-1]
            ext = p.inline_data.mime_type.split("/")[-1].replace("jpeg", "jpg")
            f = d / f"{tag}_p{i}{'' if final else '_draft'}.{ext}"
            f.write_bytes(p.inline_data.data)
            w, h = Image.open(f).size
            img = {"file": str(f.relative_to(OUT)), "mime": p.inline_data.mime_type,
                   "thought_flag": p.thought, "final": final, "size": [w, h]}
            if test == "T1_kasbah_thinking" or "kasbah" in tag:
                img["surface_changed"] = surface_stability(f)
            rec["images"].append(img)
        elif p.text:
            rec["texts"].append({"thought": p.thought, "text": p.text[:300]})
    u = resp.usage_metadata
    rec["usage"] = {"prompt": u.prompt_token_count, "output": u.candidates_token_count,
                    "thoughts": u.thoughts_token_count}
    gm = getattr(cand, "grounding_metadata", None)
    if gm:
        rec["grounding"] = {"queries": gm.web_search_queries,
                            "n_sources": len(gm.grounding_chunks or [])}
    print(f"[{test}/{tag}] {rec['wall_s']}s imgs={len(rec['images'])} "
          f"drafts={sum(not i['final'] for i in rec['images'])} usage={rec['usage']}")
    return rec


def big_noise_png(mb: int) -> bytes:
    side = int((mb * 1024 * 1024 / 3) ** 0.5)
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 256, (side, side, 3), np.uint8)).save(buf, "PNG", compress_level=0)
    return buf.getvalue()


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # FAL baselines for the same prompts (from the 2026-10-09 shootout).
    fal = OUT / "FAL_baseline"
    fal.mkdir(exist_ok=True)
    for step in ("A5", "A1"):
        for f in sorted((SHOOTOUT / step / "nb-2.1").glob("*.jpg")):
            if not f.stem.endswith("_diff"):
                shutil.copy2(f, fal / f"fal_{f.name}")

    jobs = [
        *[call("T1_kasbah_thinking", f"kasbah_{lvl}", A5_PROMPT, [KASBAH, CHARACTER],
               aspect="16:9", thinking=lvl, seed=4242,
               note="shootout A5 (WZRD-critical); compare FAL_baseline/fal_A5_*")
          for lvl in ("minimal", "medium", "high")],
        call("T2_two_refs_2K", "compose_2K", A1_PROMPT, [CHARACTER, BUILDING],
             aspect="16:9", size="2K", thinking="medium", seed=4242,
             note="shootout A1; compare FAL_baseline/fal_A1_*"),
        *[call("T3_seed", tag, "The creature from image_1 crouching low, ready to pounce. "
               f"{STYLE_LOCK} Same dark sepia paper background.", [CHARACTER],
               aspect="1:1", thinking="minimal", seed=s)
          for tag, s in (("seed7_a", 7), ("seed7_b", 7), ("seed8", 8))],
        call("T4_search", "weather_poster",
             "A minimalist poster showing today's actual weather forecast for Ghent, "
             "Belgium (temperature and conditions), with today's date.", [],
             aspect="3:4", thinking="minimal", search=True),
        call("T5_ref_limit", "refs14",
             "Arrange the 14 creatures from image_1 … image_14 in a 7×2 grid on black, "
             "in order, each labelled with its number.", CREATURES[:14],
             aspect="16:9", thinking="minimal"),
        call("T5_ref_limit", "refs15", "Arrange all 15 images in a 5×3 grid on black.",
             [*CREATURES[:14], CHARACTER], aspect="16:9", thinking="minimal"),
        call("T6_4K", "lantern_4K", "A single red paper lantern glowing on pure black, "
             "intricate paper texture.", [], aspect="16:9", size="4K", thinking="minimal"),
        call("T7_vertex_medium", "kasbah_vertex", A5_PROMPT, [KASBAH, CHARACTER],
             client=VERTEX, aspect="16:9", thinking="medium", seed=4242,
             note="check part.thought flags on Vertex drafts"),
        call("T8_payload", "noise_30MB", "Describe then redraw this as a calm sea.",
             [big_noise_png(30)], thinking="minimal", note="inline request-size limit"),
    ]
    records = await asyncio.gather(*jobs)
    (OUT / "results.json").write_text(json.dumps(records, indent=1))
    write_page(records)
    print(f"\n→ {OUT / 'index.html'}")


def write_page(records: list[dict]) -> None:
    rows = []
    fal_imgs = sorted((OUT / "FAL_baseline").glob("*.jpg"))
    rows.append("<h2>FAL baseline (shootout, nb-2.1 via FAL)</h2><div class=row>" + "".join(
        f"<figure><a href='{f.relative_to(OUT)}'><img src='{f.relative_to(OUT)}'></a>"
        f"<figcaption>{f.name}</figcaption></figure>" for f in fal_imgs) + "</div>")
    for r in records:
        meta = {k: r.get(k) for k in ("backend", "thinking", "size", "aspect", "seed",
                                      "n_refs", "wall_s", "usage", "finish_reason",
                                      "grounding", "error", "note") if r.get(k) is not None}
        figs = "".join(
            f"<figure class='{'final' if i['final'] else 'draft'}'><a href='{i['file']}'>"
            f"<img src='{i['file']}'></a><figcaption>{'FINAL' if i['final'] else 'DRAFT'} · "
            f"{i['size'][0]}×{i['size'][1]} · thought={i['thought_flag']}"
            f"{' · surface changed ' + str(i['surface_changed']) if 'surface_changed' in i else ''}"
            "</figcaption></figure>" for i in r["images"])
        texts = "".join(f"<p class=t>{'💭 ' if t['thought'] else ''}{html.escape(t['text'])}</p>"
                        for t in r["texts"])
        rows.append(f"<h2>{r['test']} / {r['tag']}</h2><pre>{html.escape(json.dumps(meta))}</pre>"
                    f"<div class=row>{figs or '<em>no image</em>'}</div>{texts}")
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Nano Banana 2.1 on Google</title><style>"
        "body{background:#111;color:#ddd;font:14px system-ui;margin:24px}"
        ".row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;max-width:520px}"
        "img{max-width:520px;max-height:360px;border:2px solid #444}"
        ".final img{border-color:#3a3}.draft img{border-color:#c63}"
        "pre{white-space:pre-wrap;color:#9ab}.t{color:#aaa;max-width:1000px}</style>"
        "<h1>Nano Banana 2.1 — direct Google probe (2026-10-09)</h1>"
        "<p>Green border = final image, orange = draft (intermediate image while thinking). "
        "‘surface changed’ = share of pixels changed vs the darkened kasbah (lower is better).</p>"
        + "".join(rows))


if __name__ == "__main__":
    asyncio.run(main())
