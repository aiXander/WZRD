// Studio room state (studio plan A2): the open folder, its listing, the
// grid selection + keyboard focus, what plays in the monitor, and the
// mirror of the hub's Studio selection (L5 — style tray, items, target).
//
// - The **target folder** (where new renders land) is a fixed folder the
//   operator picks ("Renders land in" card, `chooseTarget`); browsing other
//   folders never changes it (changed 2026-10-24 — it used to follow the open
//   folder, S1). It's the hub selection's `target`; no target = `inbox/`.
// - The listing polls every 2 s while Studio is mounted (A1), through the
//   hub (which also indexes what's new — S3) or, hub down, the read-only
//   library.db reader. Labels are optimistic: the tile changes on the key
//   press, the hub write follows, and a failure reverts with a toast.
// - The selection mirror is written on every change (debounced 300 ms) so
//   an agent's `studio_selection()` sees what Xander sees.

import { create } from 'zustand';
import {
  addRoot,
  deleteScored,
  libraryRoots,
  getSelection,
  labelRenders,
  listFolder,
  moveFiles,
  putSelection,
  type FolderListing,
  type LabelPatch,
  type LibraryItem,
  type Root,
  type Selection,
} from '../api/hub';
import { softDelete, undoToast } from '../ui/media/trash';
import { hubUp, showName, useHub } from './hub';
import { useStore } from './store';

const LAST_DIR_KEY = 'wzrd.studio.dir';

export type ScoreRange = readonly [number, number];
export const FULL_SCORES: ScoreRange = [0, 9];
export const isFullRange = (r: ScoreRange) => r[0] <= FULL_SCORES[0] && r[1] >= FULL_SCORES[1];

type StudioState = {
  dir: string | null;
  listing: FolderListing | null;
  listError: string | null;
  /** Selected item paths, in click order (the last one is the anchor). */
  selected: string[];
  /** Keyboard cursor (an item path). */
  focus: string | null;
  /** Score range filter [min, max] over 0–9. The full range shows
   *  everything incl. unrated files; any narrower range shows only rated
   *  files inside it. */
  scoreRange: ScoreRange;
  /** What the monitor shows. */
  monitor: string | null;
  /** The hub's Studio selection (L5). */
  selection: Selection;
  /** Library roots (the home + folders indexed in place) — the tree's tops. */
  roots: Root[];
  /** Folder tree: expanded dirs → their subfolders (loaded lazily). */
  expanded: Record<string, { name: string; path: string }[]>;
  /** Multi-delete armed: how many items a second ⌫ would delete (0 = not). */
  deleteArmed: number;
  /** The full-screen viewer (ui/media/Lightbox) is open on this path. */
  lightbox: string | null;
  setFocus: (p: string | null) => void;
  setScoreRange: (r: ScoreRange) => void;
  setMonitor: (p: string | null) => void;
};

export const useStudio = create<StudioState>((set) => ({
  dir: null,
  listing: null,
  listError: null,
  selected: [],
  focus: null,
  scoreRange: FULL_SCORES,
  monitor: null,
  selection: { styles: [], items: [], target: null, project: null },
  roots: [],
  expanded: {},
  deleteArmed: 0,
  lightbox: null,
  setFocus: (focus) => set({ focus }),
  setScoreRange: (scoreRange) => set({ scoreRange }),
  setMonitor: (monitor) => set({ monitor }),
}));

const toast = (text: string, tone: 'info' | 'warn' | 'bad' | 'ok' = 'info') =>
  useStore.getState().toast(text, tone);

/** The items the grid shows, in order (the score range filter applied). */
export function visibleItems(s: Pick<StudioState, 'listing' | 'scoreRange'> = useStudio.getState()): LibraryItem[] {
  const items = s.listing?.items ?? [];
  if (isFullRange(s.scoreRange)) return items;
  const [lo, hi] = s.scoreRange;
  return items.filter((i) => i.score != null && i.score >= lo && i.score <= hi);
}

/** "3–7", "≤ 4" (min 0), "5" (one score), "all". */
export function scoreRangeLabel(r: ScoreRange): string {
  if (isFullRange(r)) return 'all';
  if (r[0] === r[1]) return `${r[0]}`;
  return r[0] <= FULL_SCORES[0] ? `≤ ${r[1]}` : `${r[0]}–${r[1]}`;
}

export function itemByPath(path: string | null): LibraryItem | undefined {
  if (!path) return undefined;
  return useStudio.getState().listing?.items.find((i) => i.path === path);
}

// ---------- folder ----------

let listSeq = 0;
let pendingTimer: ReturnType<typeof setTimeout> | undefined;
const PENDING_RELIST_MS = 400;

export async function refreshListing() {
  const dir = useStudio.getState().dir;
  if (!dir) return;
  const seq = ++listSeq;
  try {
    const listing = await listFolder(dir, hubUp());
    if (seq !== listSeq || useStudio.getState().dir !== dir) return;
    useStudio.setState((s) => mergeListing(s, listing));
    loadSelection();
    // Files still being indexed: re-list soon (one at a time), so tiles land
    // as the hub indexes them instead of on the 2 s poll.
    if (listing.pending.length && !pendingTimer) {
      pendingTimer = setTimeout(() => {
        pendingTimer = undefined;
        if (useStudio.getState().dir === dir) refreshListing();
      }, PENDING_RELIST_MS);
    }
  } catch (e) {
    if (seq === listSeq) useStudio.setState({ listError: String(e) });
  }
}

/** Keep optimistic labels the hub hasn't echoed yet (pending writes). */
function mergeListing(s: StudioState, listing: FolderListing): Partial<StudioState> {
  const items = listing.items.map((it) => {
    const p = pendingLabels.get(it.render_id);
    return p ? { ...it, ...p } : it;
  });
  const paths = new Set(items.map((i) => i.path));
  return {
    listing: { ...listing, items },
    listError: null,
    selected: s.selected.filter((p) => paths.has(p)),
    focus: s.focus && paths.has(s.focus) ? s.focus : items[0]?.path ?? null,
  };
}

export async function openDir(dir: string) {
  const prev = useStudio.getState().dir;
  if (prev !== dir) {
    clearTimeout(pendingTimer);
    pendingTimer = undefined;
    useStudio.setState({ dir, listing: null, selected: [], focus: null, listError: null });
    try {
      localStorage.setItem(LAST_DIR_KEY, dir);
    } catch {
      /* ignore */
    }
  }
  await refreshListing();
}

export function lastDir(): string | null {
  try {
    return localStorage.getItem(LAST_DIR_KEY);
  } catch {
    return null;
  }
}

export async function loadRoots() {
  try {
    useStudio.setState({ roots: await libraryRoots() });
  } catch (e) {
    console.warn('roots', e);
  }
}

const trimSlash = (p: string) => p.replace(/\/+$/, '') || '/';

/** The library root `dir` sits in (the deepest one, roots can nest). */
export function rootOf(dir: string, roots: Root[]): Root | undefined {
  const d = trimSlash(dir);
  return roots
    .filter((r) => d === trimSlash(r.path) || d.startsWith(trimSlash(r.path) + '/'))
    .sort((a, b) => b.path.length - a.path.length)[0];
}

/** Expand the tree down to `dir` (its root and every folder above it), so
 *  the open folder is visible and highlighted; never collapses anything. */
export async function revealInTree(dir: string) {
  const root = rootOf(dir, useStudio.getState().roots);
  if (!root) return;
  const d = trimSlash(dir);
  let p = trimSlash(root.path);
  const chain: string[] = [];
  for (const seg of d.slice(p.length).split('/').filter(Boolean)) {
    chain.push(p);
    p = `${p}/${seg}`;
  }
  for (const a of chain) {
    if (useStudio.getState().expanded[a]) continue;
    try {
      const l = await listFolder(a, false); // subfolders only — no indexing
      useStudio.setState((s) => ({ expanded: { ...s.expanded, [a]: l.dirs } }));
    } catch (e) {
      console.warn('reveal', a, e);
      return;
    }
  }
}

export async function toggleTree(path: string) {
  const ex = useStudio.getState().expanded;
  if (ex[path]) {
    const { [path]: _, ...rest } = ex;
    useStudio.setState({ expanded: rest });
    return;
  }
  try {
    const l = await listFolder(path, false); // subfolders only — no indexing
    useStudio.setState((s) => ({ expanded: { ...s.expanded, [path]: l.dirs } }));
  } catch (e) {
    toast(`Can't open ${path}: ${e}`, 'bad');
  }
}

// ---------- grid selection ----------

export function clickItem(path: string, mods: { shift?: boolean; meta?: boolean }) {
  const s = useStudio.getState();
  let selected: string[];
  if (mods.meta) {
    selected = s.selected.includes(path) ? s.selected.filter((p) => p !== path) : [...s.selected, path];
  } else if (mods.shift && s.focus) {
    const vis = visibleItems(s).map((i) => i.path);
    const a = vis.indexOf(s.focus);
    const b = vis.indexOf(path);
    selected = a < 0 || b < 0 ? [path] : vis.slice(Math.min(a, b), Math.max(a, b) + 1);
  } else {
    selected = [path];
  }
  useStudio.setState({ selected, focus: path });
  writeSelection({ items: selected });
  showInMonitor(path, 0);
}

let monitorTimer: ReturnType<typeof setTimeout> | undefined;

/** Focusing a tile shows it in the monitor (images and videos). Keyboard
 *  moves settle for a beat first, so holding an arrow doesn't open a HAP
 *  decode stream per tile it passes. */
function showInMonitor(path: string, delayMs: number) {
  clearTimeout(monitorTimer);
  const apply = () => {
    const it = itemByPath(path);
    if (it && (it.kind === 'image' || it.kind === 'video') && useStudio.getState().focus === path) {
      useStudio.getState().setMonitor(path);
    }
  };
  if (delayMs <= 0) apply();
  else monitorTimer = setTimeout(apply, delayMs);
}

/** Arrow keys: move the cursor by `delta` items (± columns for up/down). */
export function moveFocus(delta: number, extend = false) {
  const s = useStudio.getState();
  const vis = visibleItems(s).map((i) => i.path);
  if (!vis.length) return;
  const at = s.focus ? vis.indexOf(s.focus) : -1;
  const next = vis[Math.max(0, Math.min(vis.length - 1, (at < 0 ? 0 : at) + delta))];
  if (extend) clickItem(next, { shift: true });
  else {
    useStudio.setState({ focus: next, selected: [next] });
    writeSelection({ items: [next] });
  }
  showInMonitor(next, 120);
}

/** What a key acts on: the selection, else the focused item. */
export function targets(): LibraryItem[] {
  const s = useStudio.getState();
  const paths = s.selected.length ? s.selected : s.focus ? [s.focus] : [];
  const byPath = new Map((s.listing?.items ?? []).map((i) => [i.path, i]));
  return paths.map((p) => byPath.get(p)).filter((i): i is LibraryItem => !!i);
}

// ---------- labels (S5) ----------

/** render id → label fields written but not yet seen in a listing. */
const pendingLabels = new Map<string, Partial<LibraryItem>>();

function patchItems(ids: Set<string>, patch: Partial<LibraryItem>) {
  useStudio.setState((s) =>
    s.listing
      ? {
          listing: {
            ...s.listing,
            items: s.listing.items.map((i) => (ids.has(i.render_id) ? { ...i, ...patch } : i)),
          },
        }
      : {}
  );
}

async function label(items: LibraryItem[], patch: LabelPatch, optimistic: Partial<LibraryItem>) {
  if (!items.length) return;
  if (!hubUp()) {
    toast('Labels need the content hub — start it from the Hub lamp', 'warn');
    return;
  }
  const ids = new Set(items.map((i) => i.render_id));
  const before = new Map(items.map((i) => [i.render_id, i]));
  ids.forEach((id) => pendingLabels.set(id, { ...pendingLabels.get(id), ...optimistic }));
  patchItems(ids, optimistic);
  try {
    await labelRenders({ ids: [...ids] }, patch);
  } catch (e) {
    toast(`Label failed: ${e}`, 'bad');
    useStudio.setState((s) =>
      s.listing
        ? { listing: { ...s.listing, items: s.listing.items.map((i) => before.get(i.render_id) ?? i) } }
        : {}
    );
  } finally {
    ids.forEach((id) => pendingLabels.delete(id));
    refreshListing();
  }
}

export const setScore = (items: LibraryItem[], score: number | null) => label(items, { score }, { score });

export function addTags(items: LibraryItem[], tags: string[]) {
  const clean = [...new Set(tags.map((t) => t.trim()).filter(Boolean))];
  if (!clean.length) return;
  // Optimistic per item: union of its tags and the new ones (same for all
  // when one item; for many, the hub's add_tags is the truth).
  const merged = [...new Set([...(items[0]?.tags ?? []), ...clean])];
  return label(items, { add_tags: clean }, items.length === 1 ? { tags: merged } : {});
}

export function removeTag(item: LibraryItem, tag: string) {
  return label([item], { remove_tags: [tag] }, { tags: item.tags.filter((t) => t !== tag) });
}

export const setNotes = (items: LibraryItem[], notes: string) =>
  label(items, { notes: notes.trim() || null }, { notes: notes.trim() || null });

// ---------- move / delete (hub only; delete never refused — S7) ----------
// Deletes are soft (show-setup G6): the hub moves the files to a trash for
// 30 min; the toast carries Undo, the grid header's Trash button lists them.

export async function moveTo(items: LibraryItem[], dest: string) {
  if (!items.length) return;
  try {
    const r = await moveFiles(
      items.map((i) => i.path),
      dest
    );
    toast(`Moved ${r.moved.length} file${r.moved.length === 1 ? '' : 's'} → ${short(dest)}`, 'ok');
  } catch (e) {
    toast(`Move failed: ${e}`, 'bad');
  }
  await refreshListing();
}

export async function deleteMedia(items: LibraryItem[]) {
  if (!items.length) return;
  const paths = items.map((i) => i.path);
  if (useStudio.getState().monitor && paths.includes(useStudio.getState().monitor!)) {
    useStudio.setState({ monitor: null });
  }
  await softDelete(paths, refreshListing);
  await refreshListing();
}

/** "Delete N–M": every rated file in the open folder inside the score range. */
export async function deleteScoredHere(range: ScoreRange, dryRun: boolean) {
  const dir = useStudio.getState().dir;
  if (!dir) return null;
  try {
    const r = await deleteScored(dir, range[0], range[1], dryRun);
    if (!dryRun) {
      const n = r.deleted?.length ?? r.count ?? 0;
      undoToast(
        `Moved ${n} file${n === 1 ? '' : 's'} scored ${scoreRangeLabel(range)} to the trash — restorable for 30 min`,
        (r as { trashed?: { id: string; path: string }[] }).trashed,
        refreshListing
      );
      await refreshListing();
    }
    return r;
  } catch (e) {
    toast(`Delete failed: ${e}`, 'bad');
    return null;
  }
}

// ---------- the hub selection (L5) ----------

let selTimer: ReturnType<typeof setTimeout> | undefined;
let selWrites = 0; // PUTs in flight — a poll mustn't read the old value back
let selPatch: Partial<Selection> = {};

function writeSelection(patch: Partial<Selection>) {
  useStudio.setState((s) => ({ selection: { ...s.selection, ...patch } }));
  selPatch = { ...selPatch, ...patch };
  clearTimeout(selTimer);
  selTimer = setTimeout(() => {
    selTimer = undefined;
    const p = selPatch;
    selPatch = {};
    if (!hubUp()) return;
    // The show renders are "made for" = the open project's folder name
    // (what the MCP tools take as `project=`).
    const project = showName();
    selWrites += 1;
    putSelection({ ...p, project })
      .catch((e) => console.warn('selection write:', e))
      .finally(() => (selWrites -= 1));
  }, 300);
}

/** Pull the hub's selection (an agent may have changed it) unless a local
 *  write is still debouncing. */
export async function loadSelection() {
  if (!hubUp() || selTimer || selWrites) return;
  try {
    const sel = await getSelection();
    if (selTimer || selWrites) return;
    useStudio.setState({ selection: sel });
  } catch (e) {
    console.warn('selection read:', e);
  }
}

/** Where renders land from now on (the hub's selection `target`). A folder
 *  outside every library root becomes a root, so its renders are indexed. */
export async function setTarget(dir: string) {
  writeSelection({ target: dir });
  try {
    const roots = await libraryRoots();
    const inside = roots.some((r) => dir === r.path || dir.startsWith(r.path.replace(/\/+$/, '') + '/'));
    if (!inside) {
      await addRoot(dir);
      loadRoots();
      toast(`Renders land in ${short(dir)} — added it to the library so they show up`, 'ok');
      return;
    }
  } catch (e) {
    console.warn('target root check:', e);
  }
  toast(`Renders land in ${short(dir)}`, 'ok');
}

export function addStyles(paths: string[]) {
  const cur = useStudio.getState().selection.styles;
  writeSelection({ styles: [...cur, ...paths.filter((p) => !cur.includes(p))] });
}

export function removeStyle(path: string) {
  writeSelection({ styles: useStudio.getState().selection.styles.filter((p) => p !== path) });
}

export function short(p: string): string {
  const root = useHub.getState().status?.media_root;
  if (root && p.startsWith(root)) return '~media' + p.slice(root.length);
  return p.replace(/^\/Users\/[^/]+/, '~');
}
