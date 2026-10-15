// Studio's folder grid (A2): static thumbnails (no animated thumbs, no
// autoplay — S10), score / tag / codec badges, tiles for running jobs that
// land in this folder (elapsed / ETA), files still being indexed, a
// "show ≤ N" filter and "delete ≤ N in this folder". Click / ⇧ / ⌘ select;
// double-click plays in the monitor. Tiles drag out as library paths (the
// style tray takes them; D1's deck slots will too).

import { memo, useEffect, useRef, useState } from 'react';
import { mediaUrl, type Job, type LibraryItem } from '../../api/hub';
import { useHub } from '../../state/hub';
import {
  clickItem,
  deleteScoredHere,
  short,
  useStudio,
  visibleItems,
} from '../../state/studio';
import { Btn } from '../../ui/primitives';

export const LIBRARY_DRAG = 'application/x-wzrd-paths';

const GRID_ID = 'studio-grid';

/** Columns the grid lays out right now (arrow ↑↓ step). */
export function gridColumns(): number {
  const el = document.getElementById(GRID_ID);
  if (!el) return 1;
  const cols = getComputedStyle(el).gridTemplateColumns.split(' ').filter(Boolean).length;
  return Math.max(1, cols);
}

export function MediaGrid() {
  const dir = useStudio((s) => s.dir);
  const listing = useStudio((s) => s.listing);
  const listError = useStudio((s) => s.listError);
  const maxScore = useStudio((s) => s.maxScore);
  const selected = useStudio((s) => s.selected);
  const focus = useStudio((s) => s.focus);
  const monitor = useStudio((s) => s.monitor);
  const deleteArmed = useStudio((s) => s.deleteArmed);
  const jobs = useHub((s) => s.jobs);
  const items = visibleItems({ listing, maxScore });
  const sel = new Set(selected);
  const running = jobs.filter((j) => j.status === 'running' && j.dest === dir);
  const focusRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    focusRef.current?.scrollIntoView({ block: 'nearest' });
  }, [focus]);

  return (
    <section className="h-full min-h-0 flex flex-col border border-line rounded-lg bg-panel">
      <header className="flex items-center gap-2 h-8 px-3 border-b border-line shrink-0 min-w-0">
        <div className="text-[11px] text-fg2 truncate min-w-0" title={dir ?? ''}>
          {dir ? short(dir) : 'No folder'}
        </div>
        <span className="text-[10px] text-fg3 num shrink-0">
          {listing ? `${listing.items.length}` : '–'}
          {listing?.pending.length ? ` · indexing ${listing.pending.length}` : ''}
          {running.length ? ` · rendering ${running.length}` : ''}
        </span>
        <div className="flex-1" />
        <ScoreFilter />
        <DeleteScored />
      </header>
      {deleteArmed > 1 && (
        <div className="px-3 py-1 text-[11px] bg-bad/15 text-bad border-b border-bad/40">
          Press ⌫ again to delete {deleteArmed} files (labels and settings are kept).
        </div>
      )}
      {listError && <div className="px-3 py-1 text-[11px] text-bad">{listError}</div>}
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <div
          id={GRID_ID}
          className="grid gap-2"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(148px, 1fr))' }}
        >
          {running.map((j) => (
            <JobTile key={j.job} job={j} />
          ))}
          {items.map((it) => (
            <Tile
              key={it.path}
              item={it}
              selected={sel.has(it.path)}
              focused={focus === it.path}
              playing={monitor === it.path}
              innerRef={focus === it.path ? focusRef : undefined}
            />
          ))}
          {listing?.pending.map((p) => (
            <div
              key={p}
              className="aspect-video rounded-md border border-dashed border-line2 flex items-center justify-center text-[10px] text-fg3 px-2 text-center break-all"
              title="Being indexed (fingerprint, probe, thumbnail)"
            >
              {p.split('/').pop()}
            </div>
          ))}
        </div>
        {listing && !items.length && !running.length && !listing.pending.length && (
          <div className="text-[12px] text-fg3 p-6 text-center">
            {maxScore != null ? `Nothing scored ≤ ${maxScore} here.` : 'No media in this folder.'}
          </div>
        )}
      </div>
    </section>
  );
}

const Tile = memo(function Tile({
  item,
  selected,
  focused,
  playing,
  innerRef,
}: {
  item: LibraryItem;
  selected: boolean;
  focused: boolean;
  playing: boolean;
  innerRef?: React.Ref<HTMLDivElement>;
}) {
  const [broken, setBroken] = useState(false);
  const thumb = item.thumb ?? (item.kind === 'image' ? item.path : null);
  const tags = [...item.tags, ...item.agent_tags.map((t) => `✦${t}`)];
  return (
    <div
      ref={innerRef}
      draggable
      onDragStart={(e) => {
        const st = useStudio.getState();
        const paths = st.selected.includes(item.path) ? st.selected : [item.path];
        e.dataTransfer.setData(LIBRARY_DRAG, JSON.stringify(paths));
        e.dataTransfer.setData('text/plain', paths.join('\n'));
        e.dataTransfer.effectAllowed = 'copy';
      }}
      onClick={(e) => clickItem(item.path, { shift: e.shiftKey, meta: e.metaKey || e.ctrlKey })}
      onDoubleClick={() => useStudio.getState().setMonitor(item.path)}
      title={`${item.name}${item.tool ? ` · ${item.tool}` : ''}${item.notes ? `\n${item.notes}` : ''}`}
      className={
        'group relative rounded-md overflow-hidden border cursor-pointer select-none bg-bg ' +
        (selected ? 'border-deck ring-1 ring-deck ' : 'border-line hover:border-line2 ') +
        (focused ? 'outline outline-1 outline-offset-2 outline-fg3' : '')
      }
    >
      <div className="aspect-video bg-black flex items-center justify-center">
        {thumb && !broken ? (
          <img
            src={mediaUrl(thumb)}
            loading="lazy"
            decoding="async"
            draggable={false}
            onError={() => setBroken(true)}
            className="w-full h-full object-contain"
          />
        ) : (
          <span className="text-[10px] text-fg3 px-2 text-center break-all">{item.codec ?? item.name}</span>
        )}
      </div>
      {item.score != null && (
        <span className="absolute top-1 left-1 num text-[13px] font-bold leading-none px-1.5 py-1 rounded bg-black/75 text-amber">
          {item.score}
        </span>
      )}
      <span
        className={
          'absolute top-1 right-1 text-[9px] font-bold tracking-wider px-1 py-[1px] rounded ' +
          (item.playable ? 'bg-ok/80 text-black' : 'bg-black/70 text-fg2')
        }
        title={item.playable ? 'The engine plays this' : 'Needs Make playable (HAP) before it can go on a deck'}
      >
        {(item.codec ?? item.kind ?? '?').toUpperCase().slice(0, 5)}
      </span>
      {playing && <span className="absolute bottom-7 right-1 text-[10px] text-deck">▶</span>}
      <div className="px-1.5 py-1 flex items-center gap-1 min-w-0">
        <span className="text-[10px] text-fg2 truncate flex-1 min-w-0">{item.name}</span>
      </div>
      {tags.length > 0 && (
        <div className="absolute bottom-7 left-1 flex gap-0.5 max-w-[85%] overflow-hidden">
          {tags.slice(0, 3).map((t) => (
            <span key={t} className="text-[9px] px-1 rounded bg-black/70 text-fg2 whitespace-nowrap">
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
});

function JobTile({ job }: { job: Job }) {
  const elapsed = job.elapsed_s ?? 0;
  const eta = job.eta_s;
  const frac = eta != null && elapsed + eta > 0 ? elapsed / (elapsed + eta) : null;
  return (
    <div
      className="relative rounded-md border border-deck/50 bg-bg overflow-hidden"
      title={`${job.tool} by ${job.by} — ${JSON.stringify(job.args).slice(0, 300)}`}
    >
      <div className="aspect-video flex flex-col items-center justify-center gap-1">
        <span className="text-[11px] text-deck font-semibold animate-pulse">rendering</span>
        <span className="num text-[10px] text-fg3">
          {Math.round(elapsed)} s{eta != null ? ` · ~${Math.round(eta)} s left` : ''}
        </span>
      </div>
      {frac != null && <div className="absolute bottom-0 left-0 h-[2px] bg-deck" style={{ width: `${frac * 100}%` }} />}
      <div className="px-1.5 py-1 text-[10px] text-fg2 truncate">
        {job.tool} · {job.by}
      </div>
    </div>
  );
}

function ScoreFilter() {
  const maxScore = useStudio((s) => s.maxScore);
  return (
    <label className="flex items-center gap-1 text-[10px] text-fg3 shrink-0" title="Show only media scored ≤ N">
      show
      <select
        value={maxScore ?? ''}
        onChange={(e) => useStudio.getState().setMaxScore(e.target.value === '' ? null : Number(e.target.value))}
        className="h-6 bg-bg border border-line rounded text-[11px] text-fg px-1"
      >
        <option value="">all</option>
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
          <option key={n} value={n}>
            ≤ {n}
          </option>
        ))}
      </select>
    </label>
  );
}

/** "Delete ≤ N in this folder": dry run first, then a second click. */
function DeleteScored() {
  const maxScore = useStudio((s) => s.maxScore);
  const [armed, setArmed] = useState<{ n: number; count: number } | null>(null);
  useEffect(() => setArmed(null), [maxScore]);
  if (maxScore == null) return null;
  return armed ? (
    <Btn
      size="sm"
      tone="danger"
      onClick={async () => {
        await deleteScoredHere(armed.n, false);
        setArmed(null);
      }}
      title="Delete the media files; their records, labels and settings stay"
    >
      Delete {armed.count} file{armed.count === 1 ? '' : 's'}?
    </Btn>
  ) : (
    <Btn
      size="sm"
      tone="ghost"
      onClick={async () => {
        const r: any = await deleteScoredHere(maxScore, true);
        const count = r?.would_delete?.length ?? r?.count ?? r?.files?.length ?? 0;
        setArmed({ n: maxScore, count });
      }}
      title="Delete every file scored ≤ N in this folder (unrated files are never touched)"
    >
      Delete ≤ {maxScore}…
    </Btn>
  );
}
