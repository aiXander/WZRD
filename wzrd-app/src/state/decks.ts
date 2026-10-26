// Clip decks (studio plan D1, contract L6) — Resolume's stacks of clips,
// per show, in `<project>/decks.json` (replaces the gig's clipgrid.json).
//
// - Rows = the show's regions + `_unassigned`; columns = slots (a `null`
//   is an empty slot). One deck is shown at a time; switching decks never
//   changes what plays (S19).
// - Slots reference library files in place by absolute path (+ render id
//   + fingerprint, which Relink matches on). Nothing is copied.
// - Click a slot = bind it on DESIGN (`clips.ts::bindClip`, through
//   sceneCommit); T takes it live as always.
// - Only the app writes it: whole-doc writes with a compare-and-swap on the
//   file's mtime — a "changed" reply means another seat (webview, browser
//   seat) wrote in between: re-read, re-apply, retry once. While a deck grid
//   is mounted the file is re-read every 2 s.
// - Implicit signals (S6) go to the hub: `deck_add` on add, `played` with
//   seconds on the wall when a clip leaves LIVE.

import { useEffect } from 'react';
import { create } from 'zustand';
import { call } from '../api/transport';
import { hubCall, listFolder, type LibraryItem } from '../api/hub';
import { hubUp } from './hub';
import { clipsByRegion } from './clips';
import { commitSceneMutation, settleCommits } from './sceneCommit';
import { useStore } from './store';

export const UNASSIGNED = '_unassigned';

export type Slot = { path: string; id: string | null; fp: string | null } | null;
export type Deck = { name: string; rows: Record<string, Slot[]> };
export type DecksDoc = { schema: 1; active: string; decks: Deck[] };

type DecksState = {
  /** scene_path the doc belongs to. */
  project: string | null;
  doc: DecksDoc | null;
  mtime: number | null;
  /** Slot paths whose file is gone (D3's Relink fixes them). */
  missing: Set<string>;
  error: string | null;
  /** Bumped when files were (re)indexed — slot thumbnails refetch. */
  indexed: number;
};

export const useDecks = create<DecksState>(() => ({
  project: null,
  doc: null,
  mtime: null,
  missing: new Set(),
  error: null,
  indexed: 0,
}));

type ReadReply = { doc: DecksDoc; mtime_ms: number | null; missing: string[] };

const toast = (t: string, tone: 'info' | 'warn' | 'bad' | 'ok' = 'info') => useStore.getState().toast(t, tone);

async function loadDecks() {
  const project = useStore.getState().project?.scene_path ?? null;
  if (!project) return;
  try {
    const r = await call<ReadReply>('decks_read');
    if (useStore.getState().project?.scene_path !== project) return;
    const cur = useDecks.getState();
    if (cur.project === project && cur.mtime === r.mtime_ms && cur.doc) {
      // Unchanged file; refresh the missing set only if it moved.
      if (r.missing.length !== cur.missing.size || r.missing.some((p) => !cur.missing.has(p))) {
        useDecks.setState({ missing: new Set(r.missing) });
      }
      return;
    }
    useDecks.setState({ project, doc: r.doc, mtime: r.mtime_ms, missing: new Set(r.missing), error: null });
  } catch (e) {
    useDecks.setState({ error: String(e) });
  }
}

/** Apply `op` to the doc and write it (CAS on mtime; one retry on a
 *  concurrent write). Optimistic: the grid shows the result at once. */
async function editDecks(op: (d: DecksDoc) => DecksDoc | void): Promise<boolean> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const { doc, mtime } = useDecks.getState();
    if (!doc) return false;
    const next = structuredClone(doc);
    const r = op(next);
    const out = r ?? next;
    useDecks.setState({ doc: out });
    try {
      const w = await call<{ mtime_ms: number; missing: string[] }>('decks_write', {
        doc: out,
        baseMtimeMs: mtime,
      });
      useDecks.setState({ mtime: w.mtime_ms, missing: new Set(w.missing) });
      return true;
    } catch (e) {
      if (String(e) === 'changed' && attempt === 0) {
        await loadDecks();
        continue;
      }
      toast(`Deck edit failed: ${e}`, 'bad');
      await loadDecks();
      return false;
    }
  }
  return false;
}

/** Load each show's decks when it opens (App). */
export function useDecksLoader() {
  useEffect(() => {
    loadDecks();
    return useStore.subscribe((s, prev) => {
      if (s.project?.scene_path !== prev.project?.scene_path) {
        useDecks.setState({ project: null, doc: null, mtime: null, missing: new Set() });
        loadDecks();
      }
    });
  }, []);
}

let mounted = 0;
let timer = 0;
/** Re-read decks.json every 2 s while a deck grid is on screen (another
 *  seat's edit shows up without a reload). */
export function useDecksLivePoll() {
  useEffect(() => {
    if (mounted++ === 0) {
      loadDecks();
      timer = window.setInterval(() => !document.hidden && loadDecks(), 2000);
    }
    return () => {
      if (--mounted === 0) window.clearInterval(timer);
    };
  }, []);
}

export function activeDeck(doc: DecksDoc | null): Deck | null {
  return doc?.decks.find((d) => d.name === doc.active) ?? doc?.decks[0] ?? null;
}

// ---------- deck tabs ----------

export const setActive = (name: string) =>
  editDecks((d) => {
    d.active = name;
  });

export function createDeck() {
  const doc = useDecks.getState().doc;
  if (!doc) return;
  let n = doc.decks.length + 1;
  while (doc.decks.some((d) => d.name === `Deck ${n}`)) n++;
  const name = `Deck ${n}`;
  return editDecks((d) => {
    d.decks.push({ name, rows: { [UNASSIGNED]: [] } });
    d.active = name;
  });
}

export function renameDeck(from: string, to: string) {
  const name = to.trim();
  const doc = useDecks.getState().doc;
  if (!name || name === from || !doc) return;
  if (doc.decks.some((d) => d.name === name)) {
    toast(`A deck called “${name}” exists already`, 'warn');
    return;
  }
  return editDecks((d) => {
    const deck = d.decks.find((x) => x.name === from);
    if (deck) deck.name = name;
    if (d.active === from) d.active = name;
  });
}

export function deleteDeck(name: string) {
  const doc = useDecks.getState().doc;
  if (!doc || doc.decks.length <= 1) {
    toast('A show keeps at least one deck', 'warn');
    return;
  }
  return editDecks((d) => {
    d.decks = d.decks.filter((x) => x.name !== name);
    if (d.active === name) d.active = d.decks[0].name;
  });
}

export function moveDeck(name: string, to: number) {
  return editDecks((d) => {
    const i = d.decks.findIndex((x) => x.name === name);
    if (i < 0) return;
    const [deck] = d.decks.splice(i, 1);
    d.decks.splice(Math.max(0, Math.min(d.decks.length, to)), 0, deck);
  });
}

// ---------- slots ----------

/** The row a render goes in: its region if this show's pack has it. */
function rowFor(region: string | null | undefined): string {
  const pack = useStore.getState().pack;
  return region && pack?.layers.some((l) => l.id === region) ? region : UNASSIGNED;
}

function putSlot(rows: Record<string, Slot[]>, row: string, slot: Slot, at?: number) {
  const slots = (rows[row] ??= []);
  if (at != null && at >= 0) {
    while (slots.length < at) slots.push(null);
    if (slots[at] == null) slots[at] = slot;
    else slots.splice(at, 0, slot);
    return;
  }
  const empty = slots.indexOf(null);
  if (empty >= 0) slots[empty] = slot;
  else slots.push(slot);
}

/** Add library files to the active deck — `row` (and slot `at`) or each
 *  file's own region. A non-HAP video can't play: its HAP is made first
 *  (`to_hap` job, S8) and the slot points at the `.mov` that job writes
 *  beside it — it shows as missing until the job lands. Resolves false
 *  when nothing was added. */
export async function addToDeck(
  items: Pick<LibraryItem, 'path' | 'region' | 'kind' | 'playable'>[],
  row?: string,
  at?: number
): Promise<boolean> {
  if (!items.length) return false;
  const toMake = items.filter((i) => i.kind === 'video' && !i.playable);
  const paths = items.map((i) => (toMake.includes(i) ? i.path.replace(/\.[^./]+$/, '.mov') : i.path));
  for (const it of toMake) {
    if (!hubUp()) {
      toast('Making a clip playable needs the content hub', 'warn');
      return false;
    }
    await hubCall('POST', '/api/render', { tool: 'to_hap', args: { video: it.path } }).catch((e) =>
      toast(`Make playable failed: ${e}`, 'bad')
    );
  }
  let slots: Slot[];
  try {
    slots = await call<Slot[]>('deck_slots', { paths });
  } catch (e) {
    toast(`Add to deck failed: ${e}`, 'bad');
    return false;
  }
  const ok = await editDecks((d) => {
    const deck = activeDeck(d);
    if (!deck) return;
    slots.forEach((s, i) => {
      const r = row ?? rowFor(items[i].region);
      const already = Object.values(deck.rows).some((ss) => ss.some((x) => x?.path === s?.path));
      if (!already) putSlot(deck.rows, r, s, at != null ? at + i : undefined);
    });
  });
  if (ok) {
    if (toMake.length) toast(`Making ${toMake.length} clip${toMake.length === 1 ? '' : 's'} playable — the slot lights up when the HAP lands`, 'info');
    for (const p of paths) signal(p, 'deck_add');
  }
  return ok;
}

/** Drag a slot to another row / position in the same deck. */
export function moveSlot(path: string, row: string, at?: number) {
  return editDecks((d) => {
    const deck = activeDeck(d);
    if (!deck) return;
    let slot: Slot = null;
    for (const ss of Object.values(deck.rows)) {
      const i = ss.findIndex((s) => s?.path === path);
      if (i >= 0) {
        slot = ss[i];
        ss[i] = null;
      }
    }
    if (!slot) return;
    for (const r of Object.keys(deck.rows)) {
      while (deck.rows[r].length && deck.rows[r][deck.rows[r].length - 1] == null) deck.rows[r].pop();
    }
    putSlot(deck.rows, row, slot, at);
  });
}

export function removeSlot(path: string) {
  return editDecks((d) => {
    const deck = activeDeck(d);
    if (!deck) return;
    for (const r of Object.keys(deck.rows)) {
      deck.rows[r] = deck.rows[r].map((s) => (s?.path === path ? null : s));
      while (deck.rows[r].length && deck.rows[r][deck.rows[r].length - 1] == null) deck.rows[r].pop();
    }
  });
}

// ---------- D3: Relink missing ----------

type RelinkReply = {
  missing: number;
  found: number;
  not_found: string[];
  /** Old `image.clip` string → new absolute path, for the design scene. */
  scene: Record<string, string>;
  decks: number;
  looks: number;
  /** Folders the files were found in (indexed by the hub afterwards). */
  dirs: string[];
};

/** Find moved clips (render id → fingerprint → a size walk of the roots;
 *  `relink.rs`), rewrite decks + looks in the shell, and the design scene
 *  here through sceneCommit. The wall's LIVE leg picks the new paths up
 *  on the next TAKE. */
export async function relinkMissing(): Promise<void> {
  await settleCommits();
  toast('Looking for missing clips…', 'info');
  let r: RelinkReply;
  try {
    r = await call<RelinkReply>('relink_missing', { scene: useStore.getState().sceneJson || null });
  } catch (e) {
    toast(`Relink failed: ${e}`, 'bad');
    return;
  }
  const remap = r.scene;
  if (Object.keys(remap).length) {
    commitSceneMutation((s) => {
      const walk = (v: any): void => {
        if (Array.isArray(v)) v.forEach(walk);
        else if (v && typeof v === 'object')
          for (const [k, x] of Object.entries(v)) {
            if (k === 'clip' && typeof x === 'string' && remap[x]) v[k] = remap[x];
            else walk(x);
          }
      };
      walk(s);
      return s;
    });
  }
  // Index where they turned up, so the slots get thumbnails + labels back.
  if (hubUp()) await Promise.all(r.dirs.map((d) => listFolder(d, true).catch(() => {})));
  await loadDecks();
  useDecks.setState((s) => ({ indexed: s.indexed + 1 }));
  if (!r.missing) toast('Nothing is missing', 'ok');
  else if (!r.not_found.length) toast(r.found === 1 ? 'Relinked the missing clip' : `Relinked all ${r.found} missing clips`, 'ok');
  else
    toast(
      `Relinked ${r.found} of ${r.missing} — not found: ${r.not_found.map((p) => p.split('/').pop()).join(', ')}`,
      r.found ? 'warn' : 'bad'
    );
}

// ---------- S6 implicit signals ----------

function signal(path: string, kind: 'deck_add' | 'played', seconds?: number) {
  if (!hubUp()) return;
  hubCall('POST', '/api/library/signal', { path, kind, ...(seconds != null ? { seconds } : {}) }).catch(() => {});
}

/** Absolute form of a binding's `image.clip` (C5 paths may still be
 *  relative to the scene dir in older scenes). */
export function absClip(clip: string): string {
  if (clip.startsWith('/')) return clip;
  const dir = useStore.getState().project?.dir ?? '';
  return `${dir.replace(/\/+$/, '')}/${clip}`;
}

/** App-level: log `played` (seconds on the wall) whenever a clip leaves
 *  the LIVE leg. Reads the live scene the store already mirrors. */
export function usePlayedSignals() {
  useEffect(() => {
    const since = new Map<string, number>();
    const liveSet = (json: string) =>
      new Set(Object.values(clipsByRegion(json)).flat().map(absClip));
    let cur = liveSet(useStore.getState().liveSceneJson);
    const now = Date.now();
    cur.forEach((p) => since.set(p, now));
    return useStore.subscribe((s, prev) => {
      if (s.liveSceneJson === prev.liveSceneJson) return;
      const next = liveSet(s.liveSceneJson);
      const t = Date.now();
      // A project switch or an engine restart blanks / swaps the live scene:
      // nothing "left the wall" — restart the clocks without a signal.
      if (!s.liveSceneJson || !prev.liveSceneJson || s.project?.scene_path !== prev.project?.scene_path) {
        since.clear();
        next.forEach((p) => since.set(p, t));
        cur = next;
        return;
      }
      for (const p of cur) {
        if (!next.has(p)) {
          const start = since.get(p);
          if (start) signal(p, 'played', Math.round((t - start) / 100) / 10);
          since.delete(p);
        }
      }
      for (const p of next) if (!since.has(p)) since.set(p, t);
      cur = next;
    });
  }, []);
}

