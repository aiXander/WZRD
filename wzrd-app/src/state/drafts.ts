// Draft → pick → finish in the Studio grid (render queue plan, lane U item 4):
// which tiles are TextureFlow drafts (the DRAFT badge), and Upscale on any
// number of them (key U, the grid's "Upscale N", RecordPanel's button).
//
// A draft is a texture_flow render whose `record.textureflow.stage` is
// `draft`. A listing item carries that as `item.tf` once the hub sends it;
// until then (and from the read-only library.db reader) the record is read
// once per render id and cached — the stage of a render never changes.
//
// Upscale = `texture_flow({finish_from: request_id})` — the hub runs it on
// the backend that rendered the draft (its bundle lives there, kept 24 h).
// Past `draft.expires_at` it re-renders the draft exactly instead (same args
// + seed + `draft`), to be upscaled afterwards.

import { create } from 'zustand';
import { libraryRecord, startRender, type LibraryItem, type RenderRecord, type TfBrief } from '../api/hub';
import { refreshJobs, useHub } from './hub';
import { useStore } from './store';

type DraftState = { briefs: Record<string, TfBrief | null> };

export const useDrafts = create<DraftState>(() => ({ briefs: {} }));

const loading = new Set<string>();

function briefFromRecord(rec: RenderRecord | null): TfBrief | null {
  const tf = rec?.record?.textureflow as
    | { stage?: string; request_id?: string; backend?: string; draft?: { expires_at?: string } | null }
    | undefined;
  if (!tf) return null;
  return {
    stage: tf.stage ?? null,
    request_id: tf.request_id ?? null,
    backend: tf.backend === 'local' ? 'local' : tf.backend === 'modal' ? 'modal' : null,
    expires_at: tf.draft?.expires_at ?? null,
  };
}

/** Read the texture_flow records this listing doesn't brief yet (4 at a time). */
export async function loadBriefs(items: LibraryItem[]) {
  const have = useDrafts.getState().briefs;
  const todo = [
    ...new Set(
      items
        .filter((i) => i.tool === 'texture_flow' && i.tf === undefined && i.render_id)
        .map((i) => i.render_id)
        .filter((id) => !(id in have) && !loading.has(id))
    ),
  ];
  if (!todo.length) return;
  todo.forEach((id) => loading.add(id));
  const got: Record<string, TfBrief | null> = {};
  const worker = async () => {
    for (let id = todo.shift(); id; id = todo.shift()) {
      got[id] = briefFromRecord(await libraryRecord(id).catch(() => null));
      loading.delete(id);
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]);
  useDrafts.setState((s) => ({ briefs: { ...s.briefs, ...got } }));
}

export function briefOf(item: LibraryItem, briefs: Record<string, TfBrief | null>): TfBrief | null {
  if (item.tf !== undefined) return item.tf;
  return briefs[item.render_id] ?? null;
}

/** Hours left before a draft bundle expires (negative = expired). */
export function hoursLeft(expiresAt: string | null | undefined): number {
  const t = expiresAt ? Date.parse(expiresAt) : NaN;
  return Number.isFinite(t) ? (t - Date.now()) / 3.6e6 : -1;
}

export const isDraft = (b: TfBrief | null) => b?.stage === 'draft';

/** A finish of this draft is already queued or running. */
function upscaling(requestId: string | null | undefined): boolean {
  if (!requestId) return false;
  return useHub
    .getState()
    .jobs.some((j) => (j.status === 'queued' || j.status === 'running') && j.args?.finish_from === requestId);
}

/** Finish one draft from its record: Upscale while its bundle lives, else re-draft. */
export async function finishDraft(rec: RenderRecord): Promise<'upscale' | 'redraft' | 'skip'> {
  const tf = rec.record.textureflow as { request_id?: string; draft?: { expires_at?: string } | null };
  const live = !!tf?.request_id && hoursLeft(tf.draft?.expires_at) > 0;
  if (live && upscaling(tf.request_id)) return 'skip';
  // Expired: the same args + seed as a new draft (the same first pass), finished afterwards.
  const { model: _model, ...again } = (rec.job_args ?? rec.record.args ?? {}) as Record<string, unknown>;
  const args = live ? { finish_from: tf.request_id } : { ...again, seed: rec.record.args?.seed, draft: true };
  await startRender('texture_flow', args);
  return live ? 'upscale' : 'redraft';
}

/** Upscale every draft among `items` (key U / "Upscale N"); non-drafts are ignored. */
export async function upscaleDrafts(items: LibraryItem[]) {
  const toast = useStore.getState().toast;
  const ids = [...new Set(items.filter((i) => i.tool === 'texture_flow').map((i) => i.render_id))];
  const n = { upscale: 0, redraft: 0, skip: 0, failed: 0 };
  let error = '';
  for (const id of ids) {
    try {
      const rec = await libraryRecord(id);
      if (rec.record?.textureflow?.stage !== 'draft') continue;
      n[await finishDraft(rec)] += 1;
    } catch (e) {
      n.failed += 1;
      error = String(e);
    }
  }
  if (n.upscale || n.redraft) refreshJobs();
  const parts = [
    n.upscale && `Upscaling ${n.upscale} draft${n.upscale === 1 ? '' : 's'}`,
    n.redraft && `re-drafting ${n.redraft} expired`,
    n.skip && `${n.skip} already upscaling`,
  ].filter(Boolean);
  if (n.failed) toast(`Upscale refused${n.failed > 1 ? ` (${n.failed})` : ''}: ${error}`, 'bad');
  else if (!parts.length) toast('No drafts selected — U upscales TextureFlow drafts', 'warn');
  if (parts.length) toast(parts.join(' · '), n.upscale || n.redraft ? 'ok' : 'info');
}
