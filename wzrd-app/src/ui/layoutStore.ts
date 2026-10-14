// Panel layout state for the rooms (`ui/layoutSplit.tsx` renders it):
// split sizes, pane order, and the lock. Per project, persisted by the
// shell to `<project>/ui_layout.json` (src-tauri/src/layout.rs — why not
// project.json is explained there).
//
//   {"schema": 1, "locked": true,
//    "sizes": {"<split id>/<pane id>": px, …},
//    "order": {"<split id>": ["<pane id>", …], …}}
//
// Absent keys mean "the room's default" (each <Pane> carries its default
// size), so a new pane or room needs no migration. Locked by default:
// nothing moves by accident mid-gig. Writes are debounced and name the
// project they were made in, so a project switch can't misfile them.
// Nothing here subscribes to telemetry.

import { create } from 'zustand';
import { uiLayoutRead, uiLayoutWrite } from '../api/ipc';

type LayoutDoc = {
  schema: 1;
  locked: boolean;
  sizes: Record<string, number>;
  order: Record<string, string[]>;
};

const fresh = (): LayoutDoc => ({ schema: 1, locked: true, sizes: {}, order: {} });

type LayoutState = {
  doc: LayoutDoc;
  /** Scene path the doc belongs to (null before the first project). */
  scenePath: string | null;
};

export const useLayout = create<LayoutState>(() => ({ doc: fresh(), scenePath: null }));

const SAVE_MS = 400;
let pending: { scenePath: string; doc: LayoutDoc } | null = null;
let timer = 0;

function flush() {
  window.clearTimeout(timer);
  const p = pending;
  pending = null;
  if (p) uiLayoutWrite(p.scenePath, p.doc).catch((e) => console.warn('ui_layout_write:', e));
}

function update(fn: (d: LayoutDoc) => LayoutDoc) {
  const { doc, scenePath } = useLayout.getState();
  const next = fn(doc);
  useLayout.setState({ doc: next });
  notifyLayout();
  if (!scenePath) return; // no project yet: keep it in memory only
  if (pending && pending.scenePath !== scenePath) flush();
  pending = { scenePath, doc: next };
  window.clearTimeout(timer);
  timer = window.setTimeout(flush, SAVE_MS);
}

/** Load the layout of the project at `scenePath` (hydrate calls this on
 *  every engine boot / project switch). Same project again: the in-memory
 *  doc is already the truth (the app is the file's only writer). */
export async function loadLayout(scenePath: string) {
  if (useLayout.getState().scenePath === scenePath) return;
  flush(); // the previous project's last edit goes to the previous project
  useLayout.setState({ scenePath, doc: fresh() });
  try {
    const raw = (await uiLayoutRead(scenePath)) as Partial<LayoutDoc> | null;
    if (useLayout.getState().scenePath !== scenePath) return; // switched again meanwhile
    if (raw && typeof raw === 'object') {
      useLayout.setState({
        doc: {
          schema: 1,
          locked: raw.locked !== false,
          sizes: raw.sizes && typeof raw.sizes === 'object' ? raw.sizes : {},
          order: raw.order && typeof raw.order === 'object' ? raw.order : {},
        },
      });
      notifyLayout();
    }
  } catch (e) {
    console.warn('ui_layout_read:', e);
  }
}

export function setPaneSize(key: string, px: number | null) {
  update((d) => {
    const sizes = { ...d.sizes };
    if (px == null) delete sizes[key];
    else sizes[key] = Math.round(px);
    return { ...d, sizes };
  });
}

export function setPaneOrder(split: string, ids: string[]) {
  update((d) => ({ ...d, order: { ...d.order, [split]: ids } }));
}

export function setLayoutLocked(locked: boolean) {
  update((d) => ({ ...d, locked }));
}

/** Back to the room defaults. `prefix` (a room, e.g. "perform") limits it
 *  to the splits whose id starts with `prefix.`; the lock is kept. */
export function resetLayout(prefix?: string) {
  const hit = (k: string) => !prefix || k.startsWith(prefix + '.');
  update((d) => ({
    ...d,
    sizes: Object.fromEntries(Object.entries(d.sizes).filter(([k]) => !hit(k))),
    order: Object.fromEntries(Object.entries(d.order).filter(([k]) => !hit(k))),
  }));
}

export const LAYOUT_EVENT = 'wzrd:layout';

/** Tell every `<NativeSlot>` to re-measure: a pane that moved without
 *  changing size (a swap, a neighbour's resize) fires no ResizeObserver.
 *  Coalesced to one event per frame; slots skip unchanged rects. */
let notifyRaf = 0;
export function notifyLayout() {
  if (notifyRaf) return;
  notifyRaf = requestAnimationFrame(() => {
    notifyRaf = 0;
    window.dispatchEvent(new Event(LAYOUT_EVENT));
  });
}
