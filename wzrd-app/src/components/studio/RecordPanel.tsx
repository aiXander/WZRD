// Studio's record panel (A2): what the focused item is and how it was made
// — tool, args, input styles, cost, the "made for" chip (show + region) —
// and its labels (S5: score 0–9, free tags, notes; all opt-in). Agent tags
// show with ✦ and can't be edited here (S18: they're the agent's).
//
// The record comes from library.db via the shell's read-only reader, so it
// shows while the hub is down; label edits need the hub.
//
// A TextureFlow draft (`record.textureflow.stage === 'draft'`, spec ≥ 2.2.0)
// gets Upscale: `texture_flow(finish_from=<request_id>)`, which finishes it
// from the bundle the server keeps 24 h (on the backend that rendered it —
// the hub reads that from the record). Expired → Re-draft: the same args and
// seed re-render it exactly.

import { useEffect, useMemo, useRef, useState } from 'react';
import { libraryRecord, startRender, type LibraryItem, type RenderRecord, type RenderTool } from '../../api/hub';
import { activeDeck, addToDeck, useDecks } from '../../state/decks';
import { refreshJobs } from '../../state/hub';
import { useStore } from '../../state/store';
import { addStyles, removeTag, setScore, short, targets, useStudio } from '../../state/studio';
import { Btn } from '../../ui/primitives';
import { pinAsset, unpinAsset } from '../../api/project';
import { DEFAULT_ROLES, pinsByPath, refreshProjectInfo, useProjectInfo } from '../../state/projectAssets';
import { NamePrompt } from '../../ui/media/NamePrompt';
import type { PromptKind } from './KeyPrompt';

export function RecordPanel({ onPrompt }: { onPrompt: (k: PromptKind) => void }) {
  const focus = useStudio((s) => s.focus);
  const item = useStudio((s) => s.listing?.items.find((i) => i.path === s.focus));
  const nSelected = useStudio((s) => s.selected.length);
  const hasHap = useStudio((s) =>
    item ? !!s.listing?.items.some((i) => i.render_id === item.render_id && i.codec === 'hap') : false
  );
  const [rec, setRec] = useState<RenderRecord | null>(null);
  const labelKey = item ? `${item.render_id}|${item.score}|${item.tags.join(',')}|${item.notes}` : '';

  useEffect(() => {
    if (!item?.render_id) return setRec(null);
    let live = true;
    libraryRecord(item.render_id)
      .then((r) => live && setRec(r))
      .catch(() => live && setRec(null));
    return () => {
      live = false;
    };
  }, [item?.render_id, labelKey]);

  if (!item) {
    return (
      <section className="flex-1 min-h-0 border border-line rounded-lg bg-panel p-3 text-[12px] text-fg3">
        {focus ? 'Indexing…' : 'Select an item.'}
      </section>
    );
  }
  const record = rec?.record ?? {};
  const args: Record<string, any> = record.args ?? {};
  const inputs: { name?: string; fp?: string }[] = record.inputs ?? [];
  return (
    <section className="flex-1 min-h-0 flex flex-col border border-line rounded-lg bg-panel">
      <header className="flex items-center gap-2 h-8 px-3 border-b border-line shrink-0 min-w-0">
        <div className="label truncate min-w-0" title={item.path}>
          {item.name}
        </div>
        <div className="flex-1" />
        {nSelected > 1 && <span className="text-[10px] text-deck shrink-0">{nSelected} selected — keys apply to all</span>}
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-3 flex flex-col gap-3 text-[12px]">
        <Scores item={item} />
        <ShowPins item={item} />
        <Tags item={item} onAdd={() => onPrompt('tag')} />
        <div>
          <div className="label mb-1 flex items-center">
            Notes <span className="flex-1" />
            <button className="text-[10px] text-fg3 hover:text-fg" onClick={() => onPrompt('note')}>
              edit (N)
            </button>
          </div>
          <div className="text-fg2 whitespace-pre-wrap">{item.notes || <span className="text-fg4">—</span>}</div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {item.wzrd && item.tool && <Chip>{item.tool}</Chip>}
          {(item.project || item.region) && (
            <Chip title="Made for: the show and region this render was made for">
              ⌖ {item.project ?? '—'}
              {item.region ? ` · ${item.region}` : ''}
            </Chip>
          )}
          {item.est_usd != null && <Chip>${item.est_usd.toFixed(2)}</Chip>}
          <Chip>
            {(item.codec ?? item.kind ?? '?').toUpperCase()}
            {item.w ? ` ${item.w}×${item.h}` : ''}
            {item.duration_s ? ` · ${item.duration_s.toFixed(1)} s` : ''}
          </Chip>
          {!item.wzrd && <Chip title="Not made by WZRD — identified by its fingerprint">foreign</Chip>}
        </div>
        {inputs.length > 0 && (
          <div>
            <div className="label mb-1">Inputs</div>
            {inputs.map((i, k) => (
              <div key={k} className="text-fg2 truncate" title={i.fp}>
                {i.name ?? i.fp}
              </div>
            ))}
          </div>
        )}
        {Object.keys(args).length > 0 && (
          <div>
            <div className="label mb-1">Settings</div>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
              {Object.entries(args).map(([k, v]) => (
                <div key={k} className="contents">
                  <span className="text-fg3">{k}</span>
                  <span className="text-fg2 break-all">{fmtArg(v)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {rec?.created && <div className="text-[10px] text-fg4">{rec.created} · {short(item.dir)}</div>}
      </div>
      <footer className="flex flex-wrap items-center gap-1.5 p-2 border-t border-line shrink-0">
        <Btn size="sm" onClick={() => addStyles(targets().map((t) => t.path))} title="Add to the style tray (what an agent reads as 'these styles')">
          + Styles
        </Btn>
        {item.kind === 'video' && <AddToDeck />}
        {rec?.tool === 'texture_flow' && rec.record.textureflow?.stage === 'draft' && <FinishDraft rec={rec} />}
        {rec && MORE_TOOLS.has(rec.tool ?? '') && <MoreLikeThis rec={rec} />}
        {item.kind === 'video' && !item.playable && !hasHap && <MakePlayable item={item} />}
      </footer>
    </section>
  );
}

function fmtArg(v: any): string {
  if (typeof v === 'string') return v.length > 140 ? v.slice(0, 140) + '…' : v;
  const s = JSON.stringify(v);
  return s.length > 140 ? s.slice(0, 140) + '…' : s;
}

function Chip({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <span title={title} className="text-[10px] px-1.5 py-[2px] rounded border border-line2 text-fg2 whitespace-nowrap">
      {children}
    </span>
  );
}

function Scores({ item }: { item: LibraryItem }) {
  return (
    <div>
      <div className="label mb-1">Score</div>
      <div className="flex gap-0.5">
        {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
          <button
            key={n}
            onClick={() => setScore(targets(), item.score === n ? null : n)}
            className={
              'num w-7 h-7 rounded text-[12px] font-semibold border ' +
              (item.score === n ? 'bg-amber text-black border-amber' : 'border-line text-fg3 hover:text-fg hover:border-line2')
            }
            title={item.score === n ? 'Click again to clear' : `Score ${n} (key ${n})`}
          >
            {n}
          </button>
        ))}
      </div>
    </div>
  );
}

function Tags({ item, onAdd }: { item: LibraryItem; onAdd: () => void }) {
  return (
    <div>
      <div className="label mb-1 flex items-center">
        Tags <span className="flex-1" />
        <button className="text-[10px] text-fg3 hover:text-fg" onClick={onAdd}>
          + tag (T)
        </button>
      </div>
      <div className="flex flex-wrap gap-1">
        {item.tags.map((t) => (
          <span key={t} className="group text-[11px] px-1.5 py-[1px] rounded bg-panel2 border border-line2 text-fg">
            {t}
            <button className="ml-1 text-fg4 group-hover:text-bad" onClick={() => removeTag(item, t)} title="Remove tag">
              ×
            </button>
          </span>
        ))}
        {item.agent_tags.map((t) => (
          <span key={`a-${t}`} className="text-[11px] px-1.5 py-[1px] rounded border border-dashed border-line2 text-fg3" title="Tagged by the agent">
            ✦ {t}
          </span>
        ))}
        {!item.tags.length && !item.agent_tags.length && <span className="text-fg4">—</span>}
      </div>
    </div>
  );
}


/** Tools "More like this" can re-run from a record's args. */
const MORE_TOOLS = new Set(['texture_flow', 'nano_banana', 'kling_v25_image_to_video']);

/** Same args, new seeds (S16) — N more jobs of the tool that made this. */
function MoreLikeThis({ rec }: { rec: RenderRecord }) {
  const [n, setN] = useState(4);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    // The job's own args when there is one; else the record's minus its
    // provenance-only keys. A fresh seed either way.
    // A TextureFlow finish's job args are only `finish_from`: its record holds
    // the draft's settings, which a new full render needs.
    const finish = rec.record.textureflow?.stage === 'finish';
    const src = ((finish ? rec.record.args : rec.job_args ?? rec.record.args) ?? {}) as Record<string, unknown>;
    const { seed: _seed, model: _model, finish_from: _finish, ...args } = src;
    if (rec.tool === 'nano_banana') delete args.num_images;
    setBusy(true);
    try {
      const r = await startRender(rec.tool as RenderTool, args, n);
      useStore.getState().toast(`${r.jobs.length} more like this started`, 'ok');
      refreshJobs();
    } catch (e) {
      useStore.getState().toast(`More like this refused: ${e}`, 'bad');
    } finally {
      setBusy(false);
    }
  };
  return (
    <span className="inline-flex items-center gap-0.5">
      <Btn size="sm" disabled={busy} onClick={run} title="Re-run this render's settings with new seeds">
        More like this ×{n}
      </Btn>
      <button className="text-[10px] text-fg3 hover:text-fg px-0.5" onClick={() => setN((v) => (v % 8) + 1)} title="How many (click to cycle 1–8)">
        ▾
      </button>
    </span>
  );
}

/** Hours left before a draft bundle expires (negative = expired), from its UTC `expires_at`. */
function hoursLeft(expiresAt: string | undefined): number {
  const t = expiresAt ? Date.parse(expiresAt) : NaN;
  return Number.isFinite(t) ? (t - Date.now()) / 3.6e6 : -1;
}

/** Draft → pick → finish: finish this draft (the final's first pass) from
 *  the server's bundle, or re-render it exactly once the bundle expired. */
function FinishDraft({ rec }: { rec: RenderRecord }) {
  const [busy, setBusy] = useState(false);
  const tf = rec.record.textureflow as { request_id?: string; draft?: { expires_at?: string } | null };
  const left = hoursLeft(tf.draft?.expires_at);
  const live = !!tf.request_id && left > 0;
  const run = async () => {
    // Expired: the same args + seed as a new draft (the same first pass), finished afterwards.
    const { model: _model, ...again } = (rec.job_args ?? rec.record.args ?? {}) as Record<string, unknown>;
    const args = live ? { finish_from: tf.request_id } : { ...again, seed: rec.record.args?.seed, draft: true };
    setBusy(true);
    try {
      await startRender('texture_flow', args);
      useStore.getState().toast(live ? 'Upscaling the draft' : 'Re-rendering the expired draft', 'ok');
      refreshJobs();
    } catch (e) {
      useStore.getState().toast(`${live ? 'Upscale' : 'Re-draft'} refused: ${e}`, 'bad');
    } finally {
      setBusy(false);
    }
  };
  return live ? (
    <Btn size="sm" tone="deck" disabled={busy} onClick={run} title="Finish this draft: the upscale passes only, from its saved first pass (same seed and composition)">
      Upscale · {left >= 1 ? `${Math.floor(left)} h` : `${Math.max(1, Math.round(left * 60))} min`}
    </Btn>
  ) : (
    <Btn size="sm" disabled={busy} onClick={run} title="This draft expired on the server (kept 24 h): re-render it exactly (same settings and seed), then Upscale the new one">
      Draft expired · re-draft
    </Btn>
  );
}

/** Put the selected videos on the open show's active deck (D1), each in
 *  its own region's row. A non-HAP video gets its HAP made first. Key: D. */
function AddToDeck() {
  const deck = useDecks((s) => activeDeck(s.doc)?.name);
  if (!deck) return null;
  return (
    <Btn
      size="sm"
      onClick={() => addToDeck(targets().filter((t) => t.kind === 'video'))}
      title={`Add the selected clips to the deck “${deck}” of the open show (D)`}
    >
      + Deck <span className="text-fg3">{deck}</span>
    </Btn>
  );
}

/** HAP beside the source (S8): the engine plays it; labels are shared. */
function MakePlayable({ item }: { item: LibraryItem }) {
  const [busy, setBusy] = useState(false);
  return (
    <Btn
      size="sm"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        try {
          await startRender('to_hap', { video: item.path });
          refreshJobs();
        } catch (e) {
          useStore.getState().toast(`Make playable failed: ${e}`, 'bad');
        } finally {
          setBusy(false);
        }
      }}
      title="Write a HAP .mov beside this file (same render id) so it can go on a deck and play natively"
    >
      Make playable
    </Btn>
  );
}

/** The open show's folder name (what the hub takes as `project=`). */
function useShowName(): string | null {
  const dir = useStore((s) => s.project?.dir);
  return dir ? dir.replace(/\/+$/, '').split('/').pop() ?? null : null;
}

/** Named assets of the open show that point at this file (show-setup A1),
 *  and "Pin to show as…" — e.g. pin a surface trial as `surface` so
 *  TextureFlow's shape picker and the setup wizard offer it. */
function ShowPins({ item }: { item: LibraryItem }) {
  const show = useShowName();
  const { info } = useProjectInfo(show);
  const pins = useMemo(() => pinsByPath(info)[item.path] ?? [], [info, item.path]);
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const btn = useRef<HTMLButtonElement | null>(null);
  if (!show) return null;
  const toast = useStore.getState().toast;
  const pin = async (name: string) => {
    try {
      await pinAsset(show, name, item.path);
      await refreshProjectInfo(show);
      toast(`Pinned as ${name} in ${show}`, 'ok');
    } catch (e) {
      toast(`Pin failed: ${e}`, 'bad');
    }
  };
  const unpin = async (name: string) => {
    try {
      await unpinAsset(show, name);
      await refreshProjectInfo(show);
    } catch (e) {
      toast(`Unpin failed: ${e}`, 'bad');
    }
  };
  const roles = info ? Object.keys(info.roles ?? {}) : DEFAULT_ROLES;
  return (
    <div>
      <div className="label mb-1 flex items-center">
        In show <span className="ml-1 normal-case tracking-normal text-fg3 truncate">{show}</span>
        <span className="flex-1" />
        <button
          ref={btn}
          className="text-[10px] text-fg3 hover:text-fg shrink-0"
          onClick={() => setAnchor(anchor ? null : btn.current)}
          title="Pin this file to the open show as a named asset (assets.json) — pickers and agents find it by that name"
        >
          ★ Pin to show as…
        </button>
      </div>
      <div className="flex flex-wrap gap-1">
        {pins.map((n) => (
          <span key={n} className="group text-[11px] font-mono px-1.5 py-[1px] rounded bg-amber/15 border border-amber/50 text-amber">
            ★ {n}
            <button className="ml-1 text-amber/50 group-hover:text-bad" onClick={() => unpin(n)} title={`Unpin ${n}`}>
              ×
            </button>
          </span>
        ))}
        {!pins.length && <span className="text-fg4">—</span>}
      </div>
      {anchor && (
        <NamePrompt
          anchor={anchor}
          suggestions={roles}
          initial={pins[0] ?? ''}
          title={`Pin to ${show} as…`}
          onSubmit={pin}
          onClose={() => setAnchor(null)}
        />
      )}
    </div>
  );
}
