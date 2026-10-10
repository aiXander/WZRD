"""FAL model shootout, Part A (image): Nano Banana Pro vs Nano Banana 2.1.

Results: docs/reference/hosted-models.md. Runs the same WZRD-shaped edit tasks
through both endpoints (same prompt, reference order, seed, 1K), downloads
into projects/fal-shootout/<task>/<model>/, logs every call to results.json
and writes a blind contact sheet (index.html) next to them.

    uv run python scripts/fal_shootout.py --smoke   # one call per model (~$0.20)
    uv run python scripts/fal_shootout.py           # full Part A (~$4.50)

Idempotent: a (task, model, sample) whose output exists is skipped, so a
failed or extended run only pays for the gaps.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from wzrd_mcp.fal_tools import _extract_urls, _fal_subscribe, _to_fal_url  # noqa: E402

OUT = REPO / "projects" / "fal-shootout"
STORY = REPO / "test_imgs" / "stories" / "story_01"
CHARACTER = STORY / "character.jpg"
BUILDING = STORY / "building.jpg"
KASBAH = REPO / "test_imgs" / "morocco" / "kasbah" / "input_raw_darkened.jpg"

# Pro is flat-priced; 2.1 is token-billed (~$0.034–0.044 at 1K per the plan).
MODELS = {
    "nb-pro": {"endpoint": "fal-ai/nano-banana-pro/edit", "usd": 0.15},
    "nb-2.1": {"endpoint": "google/nano-banana-2.1/edit", "usd": 0.04},
}
SAMPLES = 2
SEED0 = 4242
CONCURRENCY = 6

STYLE_LOCK = (
    "Keep the creature's exact design, proportions, glow and hand-drawn "
    "colored-pencil style."
)


# ---------------------------------------------------------------------------
# Task list. A step's refs are local paths or "@<step id>" (that step's output
# for the same model + sample), so chained steps run in order per column.
# ---------------------------------------------------------------------------
@dataclass
class Step:
    task: str
    id: str
    prompt: str
    refs: list[str]
    aspect: str = "16:9"
    label: str = ""


def _story_beat(k: int, scene: str) -> Step:
    prev = [f"@A3_beat{i}" for i in range(1, k)]
    prev_note = (
        f" Images 3–{k + 1} are the previous frames of this story: keep the same "
        "camera, framing, scene and creature exactly consistent with them."
        if prev else ""
    )
    return Step(
        "A3", f"A3_beat{k}",
        f"Frame {k} of a 4-frame story. image_1 is the creature, image_2 is the "
        f"scene, seen from the same fixed camera as image_2.{prev_note} {STYLE_LOCK} "
        f"Scene: {scene}",
        [str(CHARACTER), str(BUILDING), *prev],
        label=f"beat {k}",
    )


STEPS: list[Step] = [
    Step(
        "A1", "A1_compose",
        "Place the glowing creature from image_1 in front of the building in "
        "image_2, standing on the ground before the central doorway. "
        f"{STYLE_LOCK} Match the night lighting and color palette of image_2, "
        "and let the creature's glow cast soft light onto the stone facade.",
        [str(CHARACTER), str(BUILDING)], label="compose",
    ),
    *[
        Step(
            "A2", f"A2_{key}",
            f"The same creature from image_1, now {pose}. {STYLE_LOCK} "
            "Same dark sepia paper background as image_1.",
            [str(CHARACTER)], aspect="1:1", label=key,
        )
        for key, pose in [
            ("running", "running at full speed, seen from the side"),
            ("crouching", "crouching low to the ground, ready to pounce"),
            ("looking_back", "standing still and looking back over its shoulder"),
        ]
    ],
    _story_beat(1, "the creature peeks out from behind the bushes at the bottom "
                   "left of the dark, quiet facade."),
    _story_beat(2, "the creature creeps along the ground toward the central "
                   "doorway; its glow starts to light the stone around it."),
    _story_beat(3, "the creature rears up at the doorway, its radial crown "
                   "blazing; the two lanterns flare bright."),
    _story_beat(4, "the creature dissolves into a cloud of glowing seeds that "
                   "drift up the facade toward the clock."),
    Step(
        "A4", "A4_start",
        "The building from image_2 at night, exact same camera and framing as "
        "image_2. The creature from image_1 stands small at the bottom left, "
        f"in front of the bushes. {STYLE_LOCK}",
        [str(CHARACTER), str(BUILDING)], label="start frame",
    ),
    Step(
        "A4", "A4_end",
        "Reproduce image_3 exactly — identical camera, framing, building, "
        "lighting and creature — with one change: the creature has walked to "
        "the central doorway and now stands there facing the viewer. Change "
        "nothing else.",
        [str(CHARACTER), str(BUILDING), "@A4_start"], label="end frame",
    ),
    Step(
        "A5", "A5_projection",
        "Add the glowing creature from image_2 onto the large wall of the "
        "building in image_1, as if it were light projected onto the wall. "
        "Keep every other part of image_1 exactly unchanged: same framing, "
        "same darkness, same wall texture and sky. Only the creature is added. "
        f"{STYLE_LOCK}",
        [str(KASBAH), str(CHARACTER)], label="projection-ready",
    ),
]
STEP_BY_ID = {s.id: s for s in STEPS}
TASK_TITLES = {
    "A1": "A1 · Compose character into background",
    "A2": "A2 · Same character, new poses",
    "A3": "A3 · 4-beat story (earlier frames passed as refs)",
    "A4": "A4 · Keyframe pair for video (start → end)",
    "A5": "A5 · Projection-ready on the darkened kasbah",
}
SMOKE_STEPS = ["A1_compose"]


def out_path(step: Step, model: str, sample: int) -> Path:
    return OUT / step.task / model / f"{step.id}_s{sample}.jpg"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class Runner:
    sem: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(CONCURRENCY))
    uploads: dict[str, asyncio.Task] = field(default_factory=dict)
    done: dict[tuple[str, str, int], asyncio.Future] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)

    async def upload(self, path: str) -> str:
        if path not in self.uploads:
            self.uploads[path] = asyncio.ensure_future(_to_fal_url(path))
        return await self.uploads[path]

    async def resolve_ref(self, ref: str, model: str, sample: int) -> str:
        if ref.startswith("@"):
            local = await self.done[(ref[1:], model, sample)]
            return await self.upload(str(local))
        return await self.upload(ref)

    async def run(self, step: Step, model: str, sample: int) -> Path:
        dest = out_path(step, model, sample)
        if dest.exists():
            print(f"  skip  {step.id:16} {model:7} s{sample} (exists)")
            return dest
        urls = [await self.resolve_ref(r, model, sample) for r in step.refs]
        args = {
            "prompt": step.prompt,
            "image_urls": urls,
            "num_images": 1,
            "resolution": "1K",
            "aspect_ratio": step.aspect,
            "output_format": "jpeg",
            "seed": SEED0 + sample,
        }
        endpoint = MODELS[model]["endpoint"]
        async with self.sem:
            print(f"  call  {step.id:16} {model:7} s{sample}")
            t0 = time.time()
            result = await _fal_subscribe(endpoint, args)
            wall = time.time() - t0
        img_urls = _extract_urls(result)
        if not img_urls:
            raise RuntimeError(f"{step.id}/{model}/s{sample}: no image in {result}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.get(img_urls[0])
            r.raise_for_status()
            dest.write_bytes(r.content)
        rec = {
            "step": step.id, "task": step.task, "model": model, "sample": sample,
            "endpoint": endpoint, "args": {**args, "image_urls": step.refs},
            "seed": args["seed"], "request_id": result.get("request_id"),
            "description": result.get("description"),
            "wall_s": round(wall, 1), "est_usd": MODELS[model]["usd"],
            "output": str(dest.relative_to(OUT)),
        }
        self.records.append(rec)
        print(f"  done  {step.id:16} {model:7} s{sample}  {wall:5.1f}s")
        return dest

    async def run_tracked(self, step: Step, model: str, sample: int) -> None:
        fut = self.done[(step.id, model, sample)]
        try:
            fut.set_result(await self.run(step, model, sample))
        except Exception as exc:  # noqa: BLE001 — keep the other columns running
            print(f"  FAIL  {step.id:16} {model:7} s{sample}: {exc}")
            fut.set_exception(exc)


async def run_all(step_ids: list[str], samples: int) -> list[dict]:
    runner = Runner()
    loop = asyncio.get_running_loop()
    jobs = [(STEP_BY_ID[i], m, s) for i in step_ids for m in MODELS for s in range(samples)]
    for step, m, s in jobs:
        runner.done[(step.id, m, s)] = loop.create_future()
    await asyncio.gather(*(runner.run_tracked(*j) for j in jobs))
    for fut in runner.done.values():  # silence "exception never retrieved"
        fut.exception()
    return runner.records


# ---------------------------------------------------------------------------
# Results log + projection stability metric + contact sheet
# ---------------------------------------------------------------------------
def merge_results(new: list[dict]) -> list[dict]:
    path = OUT / "results.json"
    old = json.loads(path.read_text()) if path.exists() else []
    by_key = {(r["step"], r["model"], r["sample"]): r for r in old}
    for r in new:
        by_key[(r["step"], r["model"], r["sample"])] = r
    merged = sorted(by_key.values(), key=lambda r: (r["step"], r["model"], r["sample"]))
    path.write_text(json.dumps(merged, indent=2))
    return merged


def surface_stability(img_path: Path) -> tuple[float, Path]:
    """Share of pixels that changed vs the kasbah input (>25/255 on any
    channel), plus a diff heatmap. Lower = the surface stayed put, which is
    what an additive projector needs."""
    out = Image.open(img_path).convert("RGB")
    ref = Image.open(KASBAH).convert("RGB").resize(out.size, Image.LANCZOS)
    diff = np.abs(np.asarray(out, np.int16) - np.asarray(ref, np.int16)).max(axis=2)
    changed = float((diff > 25).mean())
    heat = img_path.with_name(img_path.stem + "_diff.png")
    Image.fromarray(np.clip(diff * 3, 0, 255).astype(np.uint8)).save(heat)
    return changed, heat


SEQUENCE_TASKS = {"A3", "A4"}  # chained steps: also vote on the whole column


def write_contact_sheet(records: list[dict]) -> Path:
    """Blind ranking page: rank the 4 cells of every row (plus a
    whole-sequence row for chained tasks), rankings kept in the browser's
    localStorage, per-model scores shown on reveal, JSON export for the
    record. Score = Borda points (1st 3, 2nd 2, 3rd 1, 4th 0) and mean rank."""
    rec_by = {(r["step"], r["model"], r["sample"]): r for r in records}
    cols = [(m, s) for m in MODELS for s in range(SAMPLES)]
    rng = random.Random(7)
    sections, n_rows = [], 0
    for task, title in TASK_TITLES.items():
        steps = [s for s in STEPS if s.task == task]
        order = list(range(len(cols)))
        rng.shuffle(order)  # one shuffle per task so chains read down a column
        blind = {cols[i]: "ABCD"[n] for n, i in enumerate(order)}
        head = "".join(
            f'<th><span class="blind">{blind[cols[i]]}</span>'
            f'<span class="real">{cols[i][0]} · s{cols[i][1]}</span></th>'
            for i in order
        )
        rows = []
        for step in steps:
            cells = []
            for i in order:
                m, s = cols[i]
                p = out_path(step, m, s)
                if not p.exists():
                    cells.append('<td class="missing">—</td>')
                    continue
                rel = p.relative_to(OUT).as_posix()
                r = rec_by.get((step.id, m, s), {})
                extra = ""
                if task == "A5":
                    changed, heat = surface_stability(p)
                    extra = (
                        f'<img class="diff" src="{heat.relative_to(OUT).as_posix()}">'
                        f'<div class="metric">{changed:.0%} of pixels changed</div>'
                    )
                meta = f'{r.get("wall_s", "?")}s' if r else ""
                cells.append(
                    f'<td class="cell" data-row="{step.id}" data-task="{task}" data-model="{m}" '
                    f'data-col="{m}|{s}"><img src="{rel}">{extra}'
                    f'<div class="meta"><a href="{rel}" target="_blank">open ↗</a> '
                    f'<span class="real">{meta}</span></div></td>'
                )
            rows.append(f'<tr><th class="rowlabel">{html.escape(step.label)}</th>{"".join(cells)}</tr>')
            n_rows += 1
        if task in SEQUENCE_TASKS:
            seq_id = f"{task}_sequence"
            cells = "".join(
                f'<td class="cell seq" data-row="{seq_id}" data-task="{task}" '
                f'data-model="{cols[i][0]}" data-col="{cols[i][0]}|{cols[i][1]}">'
                f'best whole sequence</td>'
                for i in order
            )
            rows.append(f'<tr><th class="rowlabel">sequence</th>{cells}</tr>')
            n_rows += 1
        prompts = "".join(
            f"<li><b>{html.escape(s.label)}</b>: {html.escape(s.prompt)}</li>" for s in steps
        )
        sections.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f'<table><tr><th></th>{head}</tr>{"".join(rows)}</table>'
            f"<details><summary>prompts</summary><ul>{prompts}</ul></details></section>"
        )
    total = sum(r["est_usd"] for r in records)
    walls = {
        m: [r["wall_s"] for r in records if r["model"] == m] for m in MODELS
    }
    stats = " · ".join(
        f'{m}: {len(w)} imgs, median {sorted(w)[len(w) // 2] if w else 0}s'
        for m, w in walls.items()
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAL Image Shootout</title><style>
:root{{--bg:#111;--fg:#eee;--mut:#888;--line:#333;--acc:#ffb347;--pick:#5fd38d}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0;padding:16px}}
h1{{margin:0 0 4px}} h2{{margin:28px 0 8px;font-size:16px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:1}}
.bar .btns{{display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap}}
button{{background:var(--acc);border:0;padding:6px 12px;border-radius:4px;font-weight:600;cursor:pointer}}
button.ghost{{background:transparent;color:var(--mut);border:1px solid var(--line)}}
#progress{{font-weight:600}} #progress.complete{{color:var(--pick)}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse}}
th,td{{padding:4px;vertical-align:top;text-align:left}}
th{{font-weight:600;color:var(--acc)}} .rowlabel{{color:var(--mut);white-space:nowrap}}
td.cell{{cursor:pointer;outline:3px solid transparent;outline-offset:-3px;border-radius:4px}}
td.cell:hover{{outline-color:var(--line)}}
td.cell.picked::before{{content:"#" attr(data-rank);color:var(--pick);font-weight:700;font-size:13px}}
td.cell.r1{{outline-color:var(--pick);background:#1c2a21}}
td.cell.r2,td.cell.r3,td.cell.r4{{outline-color:#3a5a46}}
td.seq{{width:300px;color:var(--mut);text-align:center;padding:10px;border:1px dashed var(--line)}}
td img{{width:300px;display:block;border-radius:3px}} td img.diff{{width:150px;margin-top:4px}}
.metric{{color:var(--acc);font-weight:600}} .meta{{color:var(--mut);font-size:12px}}
.meta a{{color:var(--mut)}}
.missing{{color:var(--mut)}} details{{color:var(--mut);font-size:12px;max-width:1240px}}
body:not(.reveal) .real{{display:none}} body.reveal .blind{{display:none}}
#scores{{display:none;margin:12px 0}} body.reveal #scores{{display:block}}
#scores table td,#scores table th{{padding:4px 12px;border-bottom:1px solid var(--line)}}
#scores .total td{{font-weight:700;color:var(--fg)}}
</style></head><body>
<div class="bar"><h1>Nano Banana Pro vs 2.1</h1>
<div class="meta">Rank each row: click images best → worst (the last one fills itself in); click a ranked image to
un-rank it. Columns are shuffled per task and labelled A–D (2 models × 2 seeds). Rankings are saved in this browser. Est. spend ${total:.2f} · {stats}</div>
<div class="btns"><span id="progress"></span>
<button id="reveal">Reveal models + scores</button>
<button class="ghost" id="export">Export rankings (JSON)</button>
<button class="ghost" id="reset">Reset rankings</button></div>
<div id="scores"></div></div>
<div class="scroll">{"".join(sections)}</div>
<script>
const KEY = "fal-shootout-ranks-v1", N_ROWS = {n_rows};
const MODELS = {json.dumps(list(MODELS))};
const POINTS = [3, 2, 1, 0];
let ranks = {{}};  // row id -> [col, col, …] best first
try {{ ranks = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(ranks)); }} catch (e) {{}} }};
const cells = [...document.querySelectorAll("td.cell")];
const rowCells = r => cells.filter(c => c.dataset.row === r);
const isDone = r => (ranks[r] || []).length === rowCells(r).length;

function tally() {{
  const by = {{}}, tasks = [];
  for (const c of cells) {{
    const label = c.dataset.row.endsWith("_sequence") ? c.dataset.task + " sequence" : c.dataset.task;
    if (!by[label]) {{ by[label] = Object.fromEntries(MODELS.map(m => [m, {{pts: 0, rankSum: 0, n: 0, firsts: 0}}])); tasks.push(label); }}
    if (!isDone(c.dataset.row)) continue;
    const k = ranks[c.dataset.row].indexOf(c.dataset.col), s = by[label][c.dataset.model];
    s.pts += POINTS[k]; s.rankSum += k + 1; s.n++; if (k === 0) s.firsts++;
  }}
  return {{ by, tasks }};
}}

function render() {{
  for (const c of cells) {{
    const k = (ranks[c.dataset.row] || []).indexOf(c.dataset.col);
    c.classList.toggle("picked", k >= 0);
    for (let i = 1; i <= 4; i++) c.classList.toggle("r" + i, k === i - 1);
    if (k >= 0) c.dataset.rank = k + 1; else delete c.dataset.rank;
  }}
  const n = [...new Set(cells.map(c => c.dataset.row))].filter(isDone).length;
  const prog = document.getElementById("progress");
  prog.textContent = `${{n}} / ${{N_ROWS}} rows ranked`;
  prog.classList.toggle("complete", n === N_ROWS);
  const {{ by, tasks }} = tally();
  const tot = Object.fromEntries(MODELS.map(m => [m, {{pts: 0, rankSum: 0, n: 0, firsts: 0}}]));
  const fmt = s => s.n ? `${{s.pts}} pts · avg rank ${{(s.rankSum / s.n).toFixed(2)}} · ${{s.firsts}}× 1st` : "–";
  let rows = "";
  for (const t of tasks) {{
    rows += `<tr><td>${{t}}</td>` + MODELS.map(m => {{
      for (const f in tot[m]) tot[m][f] += by[t][m][f];
      return `<td>${{fmt(by[t][m])}}</td>`;
    }}).join("") + "</tr>";
  }}
  rows += `<tr class="total"><td>total</td>` + MODELS.map(m => `<td>${{fmt(tot[m])}}</td>`).join("") + "</tr>";
  document.getElementById("scores").innerHTML =
    `<table><tr><th>task</th>${{MODELS.map(m => `<th>${{m}}</th>`).join("")}}</tr>${{rows}}</table>` +
    `<div class="meta">Points per row: 1st 3 · 2nd 2 · 3rd 1 · 4th 0; each model has 2 images per row. Avg rank: 1 = best, 4 = worst.` +
    (n < N_ROWS ? ` Only ${{n}} of ${{N_ROWS}} rows fully ranked so far (partial rows don't count).` : "") + `</div>`;
}}

for (const c of cells) c.addEventListener("click", e => {{
  if (e.target.closest("a")) return;
  const r = c.dataset.row, list = ranks[r] || [], k = list.indexOf(c.dataset.col);
  if (k >= 0) list.splice(k, 1);  // un-rank; the others close the gap
  else {{
    list.push(c.dataset.col);
    const left = rowCells(r).map(x => x.dataset.col).filter(x => !list.includes(x));
    if (left.length === 1) list.push(left[0]);  // last one is implied
  }}
  if (list.length) ranks[r] = list; else delete ranks[r];
  save(); render();
}});
document.getElementById("reveal").onclick = () => document.body.classList.toggle("reveal");
document.getElementById("export").onclick = () => {{
  const blob = new Blob([JSON.stringify({{ ranks, scores: tally().by, at: new Date().toISOString() }}, null, 2)],
                        {{ type: "application/json" }});
  const a = Object.assign(document.createElement("a"), {{ href: URL.createObjectURL(blob), download: "fal-shootout-ranks.json" }});
  a.click();
}};
const reset = document.getElementById("reset");
reset.onclick = () => {{  // two-step instead of a confirm() dialog
  if (reset.dataset.armed) {{ ranks = {{}}; save(); render(); delete reset.dataset.armed; reset.textContent = "Reset rankings"; }}
  else {{ reset.dataset.armed = 1; reset.textContent = "Click again to reset"; setTimeout(() => {{ delete reset.dataset.armed; reset.textContent = "Reset rankings"; }}, 3000); }}
}};
render();
</script></body></html>"""
    path = OUT / "index.html"
    path.write_text(page)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="one call per model")
    ap.add_argument("--sheet-only", action="store_true", help="rebuild index.html, no calls")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    new = []
    if not a.sheet_only:
        ids = SMOKE_STEPS if a.smoke else [s.id for s in STEPS]
        samples = 1 if a.smoke else SAMPLES
        n = len(ids) * len(MODELS) * samples
        print(f"Part A: {len(ids)} steps × {len(MODELS)} models × {samples} samples = {n} calls")
        new = asyncio.run(run_all(ids, samples))
    records = merge_results(new)
    sheet = write_contact_sheet(records)
    print(f"\n{len(new)} new images, est. ${sum(r['est_usd'] for r in new):.2f} this run")
    print(f"contact sheet: {sheet}")


if __name__ == "__main__":
    main()
