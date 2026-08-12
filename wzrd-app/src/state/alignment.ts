// §5.14 alignment commit path.
//
// Deliberately **not** routed through `sceneCommit.ts`: alignment is not scene
// content (same distinction as §5.13's identity writes), and — unlike the
// scene — the engine owns persistence outright. `alignment.json` is
// engine-written and debounced there, so this file has exactly two jobs:
//
//   1. keep the SVG under the pointer at display rate (optimistic local state)
//   2. not flood the RPC surface during a drag (one push per animation frame)
//
// There is no disk write here, no `base_rev` CAS, and no adopt step. Last
// write wins, which is the right semantics for a drag with one human editor.

import type { AlignmentDoc, AlignmentPatch, WarpPoint } from '../api/ipc';
import {
  alignmentGet,
  alignmentReset,
  alignmentSet,
  alignmentSetTestPattern,
} from '../api/ipc';
import { useStore } from './store';

/** Coalesced patch waiting for the next frame. */
let pending: AlignmentPatch | null = null;
let rafId: number | null = null;

/** Read `pending` through a call so narrowing across `await` can't lie. */
function queued(): AlignmentPatch | null {
  return pending;
}

/**
 * Which facets have an unacknowledged local edit. While a facet is in flight,
 * inbound `alignment` telemetry for it is an *older* view than what's on
 * screen, so [`ingestAlignment`] keeps the local value for that facet only.
 *
 * The corner-drag case is why this is per-facet rather than a single flag: the
 * UI pushes `corners` alone and the engine carries the extra handles for us,
 * so the echo's `points` are newer than ours and must win while its `corners`
 * are older and must not.
 */
let inflight = { corners: false, points: false };

function scheduleFlush() {
  if (rafId !== null) return;
  rafId = requestAnimationFrame(() => {
    rafId = null;
    void flushAlignment();
  });
}

/**
 * Push the coalesced patch now. Called on every animation frame during a drag
 * and explicitly on pointer-up, so the last position of a drag can't be left
 * sitting in the queue if the tab stops getting frames.
 */
export async function flushAlignment(): Promise<void> {
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  const patch = pending;
  if (!patch) return;
  pending = null;
  const sent = { corners: patch.corners !== undefined, points: patch.points !== undefined };
  try {
    const doc = await alignmentSet(patch);
    useStore.getState().setAlignmentError(null);
    // Clear only the facets this reply answers for, and only if the user
    // hasn't already queued a newer edit to them — otherwise this reply would
    // yank the handle back for a frame mid-drag.
    if (sent.corners && !queued()?.corners) inflight.corners = false;
    if (sent.points && !queued()?.points) inflight.points = false;
    ingestAlignment(doc);
  } catch (e) {
    // The engine kept rendering the previous alignment — so must the UI. Drop
    // the optimistic edit by taking the engine's document back.
    useStore.getState().setAlignmentError(String(e));
    inflight = { corners: false, points: false };
    try {
      useStore.getState().setAlignment(await alignmentGet());
    } catch {
      /* the sticky channel will re-seed it */
    }
  }
}

/**
 * Merge an engine-authored document into the store. The engine is the
 * authority — a headless camera script is as legitimate a writer as this UI —
 * so everything it says wins *except* facets with an in-flight local edit.
 */
export function ingestAlignment(doc: AlignmentDoc) {
  const s = useStore.getState();
  const local = s.alignment;
  if (!local || (!inflight.corners && !inflight.points)) {
    s.setAlignment(doc);
    return;
  }
  s.setAlignment({
    ...doc,
    corners: inflight.corners ? local.corners : doc.corners,
    points: inflight.points ? local.points : doc.points,
  });
}

function commit(patch: AlignmentPatch, optimistic?: Partial<AlignmentDoc>) {
  const s = useStore.getState();
  if (optimistic && s.alignment) s.setAlignment({ ...s.alignment, ...optimistic });
  if (patch.corners !== undefined) inflight.corners = true;
  if (patch.points !== undefined) inflight.points = true;
  pending = { ...(pending ?? {}), ...patch };
  scheduleFlush();
}

type PointPatch = NonNullable<AlignmentPatch['points']>[number];

/** The handle list in `alignment.set` shape, so callers can edit one entry. */
function patchPoints(points: WarpPoint[]): PointPatch[] {
  return points.map((p) => ({
    id: p.id,
    anchor: p.anchor,
    dest: p.dest,
    radius: p.radius,
  }));
}

// ---------- geometry ----------

/**
 * Move one corner. Pushes `corners` **alone** on purpose: that is the signal
 * the engine uses to carry the extra handles along with the content (§3.2), so
 * fine corrections stay attached to what they were correcting.
 */
export function setCorner(index: number, dest: [number, number]) {
  const cur = useStore.getState().alignment;
  if (!cur) return;
  const corners = cur.corners.map((c, i) => (i === index ? dest : c)) as [number, number][];
  commit({ corners }, { corners });
}

/**
 * Translate the whole quad — move the image on the wall without reshaping it.
 *
 * Implemented as a plain corner write rather than a new verb, which also
 * makes the extra handles ride along for free: the engine's corner-carry rule
 * keeps each handle's offset from `H(anchor)`, and for a pure translation
 * that offset is unchanged, so every local correction translates with the
 * content exactly as it should.
 *
 * `from` is the corner set the gesture started on, so a drag stays absolute
 * and can't accumulate rounding as it goes.
 */
export function panCorners(from: [number, number][], delta: [number, number]) {
  const corners = from.map(
    (c) => [c[0] + delta[0], c[1] + delta[1]] as [number, number]
  );
  commit({ corners }, { corners });
}

export function setPointDest(id: string, dest: [number, number]) {
  const cur = useStore.getState().alignment;
  if (!cur) return;
  const points = cur.points.map((p) => (p.id === id ? { ...p, dest } : p));
  commit({ points: patchPoints(points) }, { points });
}

export function setPointRadius(id: string, radius: number) {
  const cur = useStore.getState().alignment;
  if (!cur) return;
  const points = cur.points.map((p) => (p.id === id ? { ...p, radius } : p));
  commit({ points: patchPoints(points) }, { points });
}

/**
 * Drop a handle at `dest`. No anchor is sent, so the engine anchors it at the
 * current field and the rendered image does not move — adding a handle is
 * free, and so is removing one again.
 */
export async function addPoint(
  dest: [number, number],
  radius?: number
): Promise<string | null> {
  const cur = useStore.getState().alignment;
  if (!cur) return null;
  if (cur.points.length >= cur.points_max) {
    useStore
      .getState()
      .setAlignmentError(`already at the ${cur.points_max}-handle limit — remove one first`);
    return null;
  }
  // No optimistic apply: the engine assigns the id *and* the anchor (the
  // no-op-add property depends on it evaluating the current field), so there
  // is nothing meaningful to guess locally.
  const before = new Set(cur.points.map((p) => p.id));
  commit({ points: [...patchPoints(cur.points), { dest, radius }] });
  // Push now rather than at the next frame: the caller may want to drag the
  // handle it just created, and it can't until the engine names it.
  await flushAlignment();
  const after = useStore.getState().alignment;
  return after?.points.find((p) => !before.has(p.id))?.id ?? null;
}

export function removePoint(id: string) {
  const cur = useStore.getState().alignment;
  if (!cur) return;
  const points = cur.points.filter((p) => p.id !== id);
  if (useStore.getState().selectedHandle === id) {
    useStore.getState().setSelectedHandle(null);
  }
  commit({ points: patchPoints(points) }, { points });
}

// ---------- flags ----------

export function setAlignmentEnabled(enabled: boolean) {
  commit({ enabled }, { enabled });
}

export function setAlignmentBackground(background: string) {
  commit({ background }, { background });
}

export async function resetAlignment() {
  inflight = { corners: false, points: false };
  pending = null;
  ingestAlignment(await alignmentReset());
}

export async function setTestPattern(pattern: AlignmentDoc['test_pattern']) {
  ingestAlignment(await alignmentSetTestPattern(pattern));
}
