// The effect picker's library: the engine catalog plus the operator's own
// structure on top — named, foldable groups (`<effects>/library.json`, this
// module is its only writer) and per-effect thumbnails (`preview.jpg` in each
// project effect's folder, rendered by the engine's `effect.thumbnail`).
//
// A project effect no group lists is "new": that is where every effect an
// agent authors lands until the operator files it. Built-ins nobody filed sit
// in a fixed "Built-in" section. Rename and delete act on the folder on disk;
// both refuse while the LIVE leg plays the effect (live is never rewritten
// from here), and rename rewrites the design layers that use it.

import { useEffect } from 'react';
import { create } from 'zustand';
import {
  effectDescribe,
  effectLibraryRead,
  effectLibraryWrite,
  effectRemove,
  effectRename,
  effectThumbnail,
  effectThumbs,
  effectUpsert,
  listEffects,
  readEffect,
} from '../api/ipc';
import { commitSceneMutation } from './sceneCommit';
import { parseScene, prettify, type Binding } from './scene';
import { useStore, type Catalog, type Store } from './store';

export type EffectGroup = { id: string; name: string; collapsed?: boolean; effects: string[] };
export type LibraryDoc = {
  version: 1;
  groups: EffectGroup[];
  /** Fold state of the two fixed sections. */
  newCollapsed?: boolean;
  builtinCollapsed?: boolean;
};

type LibState = {
  doc: LibraryDoc | null;
  /** Scene path the doc belongs to — a project switch re-reads it. */
  docFor: string | null;
  thumbs: Record<string, string>;
};

export const useEffectLibrary = create<LibState>(() => ({ doc: null, docFor: null, thumbs: {} }));

const toast: Store['toast'] = (...a) => useStore.getState().toast(...a);

/** Pull the catalog (`effect.describe`), then the groups + thumbnails. */
export async function loadCatalog() {
  try {
    const res: any = await effectDescribe();
    const next: Catalog = {};
    for (const e of res?.effects ?? []) next[e.name] = { kind: e.kind, inputs: e.inputs ?? [] };
    useStore.getState().setCatalog(next);
  } catch (e) {
    console.warn('effect.describe', e);
    return;
  }
  await Promise.all([loadDoc(), refreshThumbs()]);
}

async function loadDoc() {
  const scenePath = useStore.getState().project?.scene_path ?? null;
  if (!scenePath || useEffectLibrary.getState().docFor === scenePath) return;
  let doc: LibraryDoc | null = null;
  try {
    const raw: any = await effectLibraryRead();
    if (raw && Array.isArray(raw.groups)) doc = { ...raw, version: 1 };
  } catch (e) {
    console.warn('effect_library_read', e);
  }
  if (!doc) {
    // First open: file what exists today under one group, so "New" starts
    // empty and only fills with what gets authored from here on.
    const users = Object.entries(useStore.getState().catalog)
      .filter(([, e]) => e.kind === 'user')
      .map(([n]) => n)
      .sort();
    doc = { version: 1, groups: users.length ? [{ id: newId(), name: 'Effects', effects: users }] : [] };
    useEffectLibrary.setState({ doc, docFor: scenePath });
    persist();
    return;
  }
  useEffectLibrary.setState({ doc, docFor: scenePath });
}

export async function refreshThumbs() {
  try {
    useEffectLibrary.setState({ thumbs: await effectThumbs() });
  } catch (e) {
    console.warn('effect_thumbs', e);
  }
}

function persist() {
  const { doc, docFor } = useEffectLibrary.getState();
  if (!doc || !docFor) return;
  effectLibraryWrite(docFor, doc).catch((e) => console.warn('effect_library_write', e));
}

function edit(fn: (d: LibraryDoc) => void) {
  const cur = useEffectLibrary.getState().doc;
  if (!cur) return;
  const d = structuredClone(cur);
  fn(d);
  useEffectLibrary.setState({ doc: d });
  persist();
}

function newId(): string {
  return 'g' + Math.random().toString(36).slice(2, 8);
}

// ---------- groups ----------

export function createGroup(name = 'New group'): string {
  const id = newId();
  edit((d) => {
    // On top, where its name field opens — not below a long list.
    d.groups.unshift({ id, name, effects: [] });
  });
  return id;
}

export function renameGroup(id: string, name: string) {
  edit((d) => {
    const g = d.groups.find((x) => x.id === id);
    if (g) g.name = name;
  });
}

/** Drop a group; its effects go back to "New". */
export function deleteGroup(id: string) {
  edit((d) => {
    d.groups = d.groups.filter((g) => g.id !== id);
  });
}

/** Fold a group, or one of the fixed sections (`new` / `builtin`). */
export function toggleGroup(id: string) {
  edit((d) => {
    if (id === 'new') d.newCollapsed = !d.newCollapsed;
    else if (id === 'builtin') d.builtinCollapsed = !d.builtinCollapsed;
    else {
      const g = d.groups.find((x) => x.id === id);
      if (g) g.collapsed = !g.collapsed;
    }
  });
}

/** File `name` under group `to` (`null` = back to New / Built-in). */
export function moveEffect(name: string, to: string | null) {
  edit((d) => {
    for (const g of d.groups) g.effects = g.effects.filter((e) => e !== name);
    const g = to ? d.groups.find((x) => x.id === to) : null;
    if (g) g.effects.push(name);
  });
}

export function groupOf(doc: LibraryDoc | null, name: string): string | null {
  return doc?.groups.find((g) => g.effects.includes(name))?.id ?? null;
}

// ---------- effects ----------

/** What the operator typed → a folder/scene name the engine accepts. */
export function toEffectId(display: string): string {
  return display
    .trim()
    .replace(/[^A-Za-z0-9-]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function layersUsing(json: string, name: string): string[] {
  return (parseScene(json)?.bindings ?? []).filter((b: Binding) => b.effect === name).map((b) => b.id);
}

function liveBlock(name: string): boolean {
  const live = layersUsing(useStore.getState().liveSceneJson, name);
  if (!live.length) return false;
  toast(`${prettify(name)} is playing on LIVE — change it once it's off the wall`, 'warn');
  return true;
}

export async function renameEffect(from: string, display: string) {
  const to = toEffectId(display);
  if (!to || to === from) return;
  if (liveBlock(from)) return;
  try {
    await effectRename(from, to);
  } catch (e) {
    toast(`Rename failed: ${e}`, 'bad');
    return;
  }
  // Design layers follow the new name (the engine keeps the old one
  // resolvable until then, so nothing goes dark in between).
  if (layersUsing(useStore.getState().sceneJson, from).length) {
    commitSceneMutation((s) => {
      for (const b of s.bindings ?? []) if (b.effect === from) b.effect = to;
      return s;
    });
  }
  edit((d) => {
    for (const g of d.groups) g.effects = g.effects.map((e) => (e === from ? to : e));
  });
  await refreshCatalogAndFiles();
}

export async function deleteEffect(name: string) {
  if (liveBlock(name)) return;
  const used = layersUsing(useStore.getState().sceneJson, name);
  if (used.length) {
    toast(`${prettify(name)} lights ${used.join(', ')} — give ${used.length === 1 ? 'that layer' : 'those layers'} another effect first`, 'warn');
    return;
  }
  let saved: { wgsl: string; descriptor: string | null } | null = null;
  try {
    saved = await readEffect(name);
  } catch {
    // Unreadable source: delete anyway, just without an undo.
  }
  const group = groupOf(useEffectLibrary.getState().doc, name);
  try {
    await effectRemove(name);
  } catch (e) {
    toast(`Delete failed: ${e}`, 'bad');
    return;
  }
  edit((d) => {
    for (const g of d.groups) g.effects = g.effects.filter((e) => e !== name);
  });
  await refreshCatalogAndFiles();
  toast(`Deleted ${prettify(name)}`, 'info', {
    action: saved
      ? {
          label: 'Undo',
          run: () => void restoreEffect(name, saved!, group),
        }
      : undefined,
  });
}

async function restoreEffect(name: string, saved: { wgsl: string; descriptor: string | null }, group: string | null) {
  try {
    await effectUpsert(name, saved.wgsl, saved.descriptor ? JSON.parse(saved.descriptor) : null);
  } catch (e) {
    toast(`Undo failed: ${e}`, 'bad');
    return;
  }
  if (group) moveEffect(name, group);
  await refreshCatalogAndFiles();
}

async function refreshCatalogAndFiles() {
  try {
    useStore.getState().setEffects(await listEffects());
  } catch (e) {
    console.warn('list_effects', e);
  }
  await loadCatalog();
}

/** A design layer that shows `name`: the selected one if it does, else the first. */
export function layerFor(name: string): string | null {
  const st = useStore.getState();
  const using = layersUsing(st.sceneJson, name);
  if (st.selectedBindingId && using.includes(st.selectedBindingId)) return st.selectedBindingId;
  return using[0] ?? null;
}

/** Re-take `name`'s thumbnail from the layer that shows it right now. */
export async function updateThumbnail(name: string, opts: { quiet?: boolean } = {}): Promise<boolean> {
  const layer = layerFor(name);
  if (!layer) {
    if (!opts.quiet) toast(`Put ${prettify(name)} on a layer first — the thumbnail is that layer's light`, 'warn');
    return false;
  }
  try {
    await effectThumbnail(layer);
  } catch (e) {
    if (!opts.quiet) toast(`Thumbnail failed: ${e}`, 'bad');
    return false;
  }
  await refreshThumbs();
  return true;
}

// ---------- automatic first thumbnail ----------

const autoTried = new Set<string>();

/** Give every project effect a design layer shows, and that has no
 *  thumbnail yet, one — once it has had a moment to animate. Called on
 *  scene/catalog changes while Prepare is open. */
export function scheduleAutoThumbs() {
  const st = useStore.getState();
  const { thumbs, docFor } = useEffectLibrary.getState();
  if (!docFor) return;
  for (const b of parseScene(st.sceneJson)?.bindings ?? []) {
    const name = b.effect;
    if (typeof name !== 'string' || st.catalog[name]?.kind !== 'user' || thumbs[name]) continue;
    const key = `${docFor}\u0000${name}`;
    if (autoTried.has(key)) continue;
    autoTried.add(key);
    window.setTimeout(() => {
      if (useEffectLibrary.getState().thumbs[name]) return;
      void updateThumbnail(name, { quiet: true });
    }, 1500);
  }
}

/** Mount where design layers are authored (Prepare): runs the auto-capture. */
export function useAutoThumbs() {
  const sceneJson = useStore((s) => s.sceneJson);
  const catalog = useStore((s) => s.catalog);
  const thumbs = useEffectLibrary((s) => s.thumbs);
  const docFor = useEffectLibrary((s) => s.docFor);
  useEffect(() => scheduleAutoThumbs(), [sceneJson, catalog, thumbs, docFor]);
}
