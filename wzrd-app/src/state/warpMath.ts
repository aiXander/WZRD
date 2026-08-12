// §5.14 warp evaluation for the Align canvas.
//
// The engine bakes the real thing into a LUT; this is the same model on the
// CPU so the canvas can *draw* the field the operator is editing. It reads
// the solved RBF coefficients straight off the `alignment` payload rather
// than re-solving — one solver, one answer, no drift between what the canvas
// shows and what the projector does.
//
// Only `makeWarp` and `warpGrid` are exported: the primitives below are the
// same maths as `alignment.rs`, and there must be exactly one place in the UI
// that assembles them or the canvas starts disagreeing with the projector.
//
// Two directions are needed and only one is closed-form:
//   toSource: dest → source   — the model itself, direct
//   toDest:   source → dest   — inverted numerically, for drawing a grid of
//                               source-space lines where they actually land

import type { AlignmentDoc } from '../api/ipc';

type Mat3 = number[]; // row-major, length 9

/**
 * Heckbert's unit-square→quad map. `corners` are the dest positions of source
 * `(0,0) (1,0) (1,1) (0,1)`. Returns **source → dest**; `null` for a
 * degenerate quad (the engine rejects those, so this is a transient state
 * mid-drag at worst).
 */
function homographyFromCorners(c: [number, number][]): Mat3 | null {
  if (c.length !== 4) return null;
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = c;
  const sx = x0 - x1 + x2 - x3;
  const sy = y0 - y1 + y2 - y3;
  let g = 0;
  let h = 0;
  if (Math.abs(sx) > 1e-12 || Math.abs(sy) > 1e-12) {
    const dx1 = x1 - x2;
    const dx2 = x3 - x2;
    const dy1 = y1 - y2;
    const dy2 = y3 - y2;
    const den = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(den) < 1e-12) return null;
    g = (sx * dy2 - dx2 * sy) / den;
    h = (dx1 * sy - sx * dy1) / den;
  }
  // prettier-ignore
  return [
    x1 - x0 + g * x1, x3 - x0 + h * x3, x0,
    y1 - y0 + g * y1, y3 - y0 + h * y3, y0,
    g,                h,                1,
  ];
}

function invert3(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const c00 = e * i - f * h;
  const c01 = f * g - d * i;
  const c02 = d * h - e * g;
  const det = a * c00 + b * c01 + c * c02;
  if (!isFinite(det) || Math.abs(det) < 1e-12) return null;
  const k = 1 / det;
  return [
    c00 * k, (c * h - b * i) * k, (b * f - c * e) * k,
    c01 * k, (a * i - c * g) * k, (c * d - a * f) * k,
    c02 * k, (b * g - a * h) * k, (a * e - b * d) * k,
  ];
}

function applyH(m: Mat3, p: [number, number]): [number, number] {
  const x = m[0] * p[0] + m[1] * p[1] + m[2];
  const y = m[3] * p[0] + m[4] * p[1] + m[5];
  let w = m[6] * p[0] + m[7] * p[1] + m[8];
  if (Math.abs(w) < 1e-9) w = w < 0 ? -1e-9 : 1e-9;
  return [x / w, y / w];
}

/** Wendland C², φ(t) = (1−t)⁴(4t+1) for t < 1 — mirrors `alignment.rs`. */
function wendland(t: number): number {
  if (t >= 1 || !isFinite(t)) return 0;
  const u = 1 - t;
  return u * u * u * u * (4 * t + 1);
}

export type Warp = {
  /** source → dest, the base quad map (what the underlay image uses). */
  h: Mat3;
  /** dest → source, the full model including local corrections. */
  toSource: (d: [number, number]) => [number, number];
  /** source → dest, inverted numerically. */
  toDest: (a: [number, number]) => [number, number];
  /** True when at least one handle actually bends the field. */
  hasResidual: boolean;
};

/**
 * Build an evaluator from an engine document. `null` if the corner quad is
 * degenerate.
 *
 * `weights` may be absent (an older engine, or a payload that predates them);
 * the base homography still draws correctly, the local corrections just don't
 * show — better than refusing to render the tab.
 */
export function makeWarp(doc: AlignmentDoc): Warp | null {
  const h = homographyFromCorners(doc.corners);
  if (!h) return null;
  const hInv = invert3(h);
  if (!hInv) return null;

  const pts = doc.points;
  const w = doc.weights ?? [];
  const n = Math.min(pts.length, w.length);
  const hasResidual = w
    .slice(0, n)
    .some(([wx, wy]) => Math.abs(wx) > 1e-7 || Math.abs(wy) > 1e-7);

  function residual(d: [number, number]): [number, number] {
    let rx = 0;
    let ry = 0;
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const dx = d[0] - p.dest[0];
      const dy = d[1] - p.dest[1];
      const f = wendland(Math.sqrt(dx * dx + dy * dy) / p.radius);
      if (f !== 0) {
        rx += w[i][0] * f;
        ry += w[i][1] * f;
      }
    }
    return [rx, ry];
  }

  function toSource(d: [number, number]): [number, number] {
    const base = applyH(hInv!, d);
    const r = residual(d);
    return [base[0] + r[0], base[1] + r[1]];
  }

  /**
   * Invert the model. Substituting `d = H(s)` turns `W(d) = a` into
   * `s = a − R(H(s))` — a fixed point in *source* space, which converges
   * quickly because the residual is a small, compactly-supported
   * perturbation. Falls back to the base quad if a pathological handle
   * arrangement refuses to settle, so the grid degrades rather than
   * disappearing.
   */
  function toDest(a: [number, number]): [number, number] {
    if (!hasResidual) return applyH(h!, a);
    let s: [number, number] = a;
    for (let i = 0; i < 16; i++) {
      const r = residual(applyH(h!, s));
      const next: [number, number] = [a[0] - r[0], a[1] - r[1]];
      const done = Math.abs(next[0] - s[0]) < 1e-7 && Math.abs(next[1] - s[1]) < 1e-7;
      s = next;
      if (done) break;
    }
    return applyH(h!, s);
  }

  return { h, toSource, toDest, hasResidual };
}

/**
 * The quad's outline in dest space, sampled. Follows the *warped* edge, not
 * the straight corner-to-corner chord — with a handle bending a side, those
 * are visibly different, and a click on the drawn edge has to land on the
 * drawn edge.
 */
export function boundarySamples(warp: Warp, perSide: number): [number, number][] {
  const out: [number, number][] = [];
  const sides: [[number, number], [number, number]][] = [
    [[0, 0], [1, 0]],
    [[1, 0], [1, 1]],
    [[1, 1], [0, 1]],
    [[0, 1], [0, 0]],
  ];
  for (const [a, b] of sides) {
    for (let i = 0; i < perSide; i++) {
      const t = i / perSide;
      out.push(warp.toDest([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]));
    }
  }
  return out;
}

/**
 * Nearest point on the sampled outline to `uv`, refined onto the segment
 * between the two closest samples so the snap is smooth rather than steppy.
 * Distances are measured in output *pixels* — a uv-space metric would bias
 * the snap toward the long axis on a non-square projector.
 */
export function snapToBoundary(
  samples: [number, number][],
  uv: [number, number],
  out: [number, number]
): [number, number] {
  const n = samples.length;
  if (n === 0) return uv;
  const [ow, oh] = out;
  const d2 = (p: [number, number]) => {
    const dx = (p[0] - uv[0]) * ow;
    const dy = (p[1] - uv[1]) * oh;
    return dx * dx + dy * dy;
  };
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < n; i++) {
    const d = d2(samples[i]);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  // Project onto whichever neighbouring segment is closer.
  let result = samples[best];
  let resultD = bestD;
  for (const j of [(best - 1 + n) % n, (best + 1) % n]) {
    const a = samples[best];
    const b = samples[j];
    const abx = (b[0] - a[0]) * ow;
    const aby = (b[1] - a[1]) * oh;
    const len2 = abx * abx + aby * aby;
    if (len2 < 1e-9) continue;
    let t = (((uv[0] - a[0]) * ow) * abx + ((uv[1] - a[1]) * oh) * aby) / len2;
    t = Math.max(0, Math.min(1, t));
    const p: [number, number] = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    const d = d2(p);
    if (d < resultD) {
      resultD = d;
      result = p;
    }
  }
  return result;
}

/**
 * Source-space grid mapped into dest space, as SVG polyline point strings.
 * This is the honest picture of the deformation: with no handles it is the
 * corner quad subdivided, and every local correction bends it exactly the way
 * the projector will.
 */
export function warpGrid(
  warp: Warp,
  cells: number,
  samples: number,
  scale: [number, number]
): string[] {
  const [sx, sy] = scale;
  const lines: string[] = [];
  const at = (u: number, v: number) => {
    const [x, y] = warp.toDest([u, v]);
    return `${(x * sx).toFixed(2)},${(y * sy).toFixed(2)}`;
  };
  for (let i = 0; i <= cells; i++) {
    const t = i / cells;
    const row: string[] = [];
    const col: string[] = [];
    for (let j = 0; j <= samples; j++) {
      const u = j / samples;
      row.push(at(u, t));
      col.push(at(t, u));
    }
    lines.push(row.join(' '), col.join(' '));
  }
  return lines;
}
