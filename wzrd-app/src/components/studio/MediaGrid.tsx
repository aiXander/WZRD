// Studio's folder grid (A2): static thumbnails (no animated thumbs, no
// autoplay — S10), score / tag / codec badges, tiles for running jobs that
// land in this folder (elapsed / ETA), files still being indexed, a score
// range filter (two thumbs over 0–9; full range = all incl. unrated) and
// "delete N–M in this folder" (to the trash, like every delete — show-setup
// G6; the header's Trash button restores). Click / ⇧ / ⌘ select, and the
// clicked (focused) tile shows in the monitor; hover ⤢ (or Space / F /
// Enter) opens the full-screen Lightbox. TextureFlow drafts carry a DRAFT badge
// (hours left on their server bundle; "↑" while a finish is queued); with
// drafts selected the header offers "Upscale N" (= key U, state/drafts.ts).
// The header's Queue button shows / hides the render queue under the grid.
// Tiles drag out as library paths (the selection when the pressed tile is in
// it) with pointer events — `ui/media/pathDrag.ts`, not HTML5 DnD, which the
// shell's native drag-drop swallows: the style tray and the Render panel's
// slots take them.

import { memo, useEffect, useRef, useState } from 'react';
import { mediaUrl, startRender, type LibraryItem, type TfBrief } from '../../api/hub';
import { refreshJobs, useHub } from '../../state/hub';
import { briefOf, hoursLeft, isDraft, loadBriefs, upscaleDrafts, useDrafts } from '../../state/drafts';
import { setQueueOpen, useQueue } from '../../state/renderQueue';
import { useStore } from '../../state/store';
import {
  clickItem,
  deleteScoredHere,
  FULL_SCORES,
  isFullRange,
  openDir,
  rootOf,
  scoreRangeLabel,
  short,
  type ScoreRange,
  useStudio,
  visibleItems,
} from '../../state/studio';
import { Btn } from '../../ui/primitives';
import { JobTile } from '../../ui/media/JobTile';
import { RecentlyDeleted } from '../../ui/media/RecentlyDeleted';
import { copyPaths } from '../../ui/media/copyPath';
import { startPathDrag } from '../../ui/media/pathDrag';
import { sameDir } from '../../ui/media/types';

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
  const scoreRange = useStudio((s) => s.scoreRange);
  const selected = useStudio((s) => s.selected);
  const focus = useStudio((s) => s.focus);
  const monitor = useStudio((s) => s.monitor);
  const deleteArmed = useStudio((s) => s.deleteArmed);
  const jobs = useHub((s) => s.jobs);
  const items = visibleItems({ listing, scoreRange });
  const sel = new Set(selected);
  const running = jobs.filter((j) => j.status === 'running' && sameDir(j.dest, dir));
  const focusRef = useRef<HTMLDivElement | null>(null);
  const briefs = useDrafts((s) => s.briefs);
  // Finishes in flight, by the draft's request id (the tile shows "↑").
  const finishing = new Set(
    jobs.filter((j) => (j.status === 'queued' || j.status === 'running') && j.args?.finish_from).map((j) => String(j.args.finish_from))
  );

  useEffect(() => {
    if (listing) loadBriefs(listing.items);
  }, [listing]);

  useEffect(() => {
    focusRef.current?.scrollIntoView({ block: 'nearest' });
  }, [focus]);

  return (
    <section className="h-full min-h-0 flex flex-col border border-line rounded-lg bg-panel">
      <header className="flex items-center gap-2 h-8 px-3 border-b border-line shrink-0 min-w-0">
        <Breadcrumb dir={dir} />
        <span className="text-[10px] text-fg3 num shrink-0">
          {listing ? `${listing.items.length}` : '–'}
          {listing?.pending.length ? ` · indexing ${listing.pending.length}` : ''}
          {running.length ? ` · rendering ${running.length}` : ''}
        </span>
        <div className="flex-1" />
        <UpscaleSelected />
        <MakeFolderPlayable />
        {/* Delete sits left of the slider: it appears mid-drag, and must not shift the slider under the pointer. */}
        <DeleteScored />
        <ScoreFilter />
        <RecentlyDeleted />
        <QueueToggle />
      </header>
      {deleteArmed > 1 && (
        <div className="px-3 py-1 text-[11px] bg-bad/15 text-bad border-b border-bad/40">
          Press ⌫ again to move {deleteArmed} files to the trash (restorable for 30 min; labels and settings are kept).
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
              draft={draftBadge(it, briefs, finishing)}
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
            {!isFullRange(scoreRange) ? `Nothing scored ${scoreRangeLabel(scoreRange)} here.` : 'No media in this folder.'}
          </div>
        )}
      </div>
    </section>
  );
}

/** The open folder as a breadcrumb: the part above its library root dimmed
 *  (not clickable), the root and every folder below it open that folder in
 *  the grid (the tree follows — FolderTree reveals the open dir). */
function Breadcrumb({ dir }: { dir: string | null }) {
  const roots = useStudio((s) => s.roots);
  if (!dir) return <div className="text-[11px] text-fg3 min-w-0">No folder</div>;
  const root = rootOf(dir, roots);
  if (!root) {
    return (
      <div className="text-[11px] text-fg2 truncate min-w-0" title={dir}>
        {short(dir)}
      </div>
    );
  }
  const rootPath = root.path.replace(/\/+$/, '');
  const parent = rootPath.slice(0, rootPath.lastIndexOf('/'));
  // The home root reads "~media" (short() of the media root); others by name.
  const dimmed = root.home ? '' : `${short(parent || '/')}/`;
  const crumbs = [{ name: root.home ? short(rootPath) : rootPath.split('/').pop() || rootPath, path: rootPath }];
  for (const seg of dir.replace(/\/+$/, '').slice(rootPath.length).split('/').filter(Boolean)) {
    crumbs.push({ name: seg, path: `${crumbs[crumbs.length - 1].path}/${seg}` });
  }
  return (
    <nav className="flex items-center min-w-[6rem] overflow-hidden whitespace-nowrap text-[11px]" title={dir}>
      {dimmed && <span className="text-fg4 truncate min-w-[1.5em] shrink">{dimmed}</span>}
      {crumbs.map((c, i) => {
        const last = i === crumbs.length - 1;
        return (
          <span key={c.path} className={'flex items-center min-w-0 ' + (last ? 'shrink' : 'shrink-0')}>
            {i > 0 && <span className="text-fg4 px-px">/</span>}
            <button
              className={
                'truncate rounded px-0.5 hover:bg-raised hover:text-fg ' + (last ? 'text-fg' : 'text-fg2')
              }
              onClick={() => !last && openDir(c.path)}
              title={last ? 'The open folder' : `Open ${short(c.path)}`}
            >
              {c.name}
            </button>
          </span>
        );
      })}
    </nav>
  );
}

/** A draft's tile badge — `↑` (a finish is queued), `⌛` (bundle expired) or
 *  the time its bundle has left ("7h") — a string, so the memoised tile
 *  only re-renders when it changes. Null = not a draft. */
type DraftBadge = string | null;

function draftBadge(item: LibraryItem, briefs: Record<string, TfBrief | null>, finishing: Set<string>): DraftBadge {
  const b = briefOf(item, briefs);
  if (!b || !isDraft(b)) return null;
  if (b.request_id && finishing.has(b.request_id)) return '↑';
  const h = hoursLeft(b.expires_at);
  return h >= 1 ? `${Math.floor(h)}h` : h > 0 ? `${Math.max(1, Math.round(h * 60))}m` : '⌛';
}

const Tile = memo(function Tile({
  item,
  selected,
  focused,
  playing,
  innerRef,
  draft,
}: {
  item: LibraryItem;
  selected: boolean;
  focused: boolean;
  playing: boolean;
  innerRef?: React.Ref<HTMLDivElement>;
  draft: DraftBadge;
}) {
  const [broken, setBroken] = useState(false);
  const thumb = item.thumb ?? (item.kind === 'image' ? item.path : null);
  const tags = [...item.tags, ...item.agent_tags.map((t) => `✦${t}`)];
  return (
    <div
      ref={innerRef}
      onPointerDown={(e) =>
        startPathDrag(e, () => {
          const st = useStudio.getState();
          const paths = st.selected.includes(item.path) ? st.selected : [item.path];
          return { paths, thumb: thumb ?? null };
        })
      }
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
      {draft && (
        <span
          className={
            'absolute top-1 text-[9px] font-bold tracking-wider px-1 py-[1px] rounded border ' +
            (item.score != null ? 'left-8 ' : 'left-1 ') +
            (draft === '⌛' ? 'bg-black/75 text-fg3 border-line2' : 'bg-black/75 text-amber border-amber/60')
          }
          title={
            draft === '↑'
              ? 'Draft — its upscale is in the render queue'
              : draft === '⌛'
              ? 'Draft — its server bundle expired; U re-renders it exactly (then upscale that one)'
              : `Draft — U upscales it (bundle kept ${draft} more)`
          }
        >
          DRAFT {draft}
        </span>
      )}
      {playing && (
        <span
          className="absolute bottom-7 right-1 flex items-center gap-1 px-1 py-[1px] rounded bg-black/75 border border-deck/60 text-[9px] font-semibold tracking-wider text-deck cursor-default"
          title="Showing in the Monitor"
        >
          <svg width="9" height="8" viewBox="0 0 9 8" fill="none" stroke="currentColor" strokeWidth="1.2" aria-hidden>
            <rect x="0.6" y="0.6" width="7.8" height="5" rx="0.8" />
            <path d="M3 7.4h3" />
          </svg>
          MONITOR
        </span>
      )}
      <button
        className="absolute top-6 right-1 w-6 h-6 rounded bg-black/75 border border-line2 text-[12px] leading-none text-fg2 hover:text-fg hover:border-fg4 opacity-0 group-hover:opacity-100 transition-opacity"
        title="Full-screen (Space / F / Enter)"
        onClick={(e) => {
          e.stopPropagation();
          clickItem(item.path, {});
          useStudio.setState({ lightbox: item.path });
        }}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        ⤢
      </button>
      <button
        className="absolute top-[52px] right-1 w-6 h-6 rounded bg-black/75 border border-line2 text-[11px] leading-none text-fg2 hover:text-fg hover:border-fg4 opacity-0 group-hover:opacity-100 transition-opacity"
        title="Copy the full path (the whole selection, one per line, when this tile is selected)"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          const sel = useStudio.getState().selected;
          copyPaths(sel.includes(item.path) ? sel : [item.path]);
        }}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        ⧉
      </button>
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

/** "Upscale N": the selected TextureFlow drafts (same as key U). */
function UpscaleSelected() {
  const selected = useStudio((s) => s.selected);
  const listing = useStudio((s) => s.listing);
  const briefs = useDrafts((s) => s.briefs);
  const sel = new Set(selected);
  const ids = new Set(
    (listing?.items ?? []).filter((i) => sel.has(i.path) && isDraft(briefOf(i, briefs))).map((i) => i.render_id)
  );
  if (!ids.size) return null;
  return (
    <Btn
      size="sm"
      tone="deck"
      onClick={() => upscaleDrafts((listing?.items ?? []).filter((i) => sel.has(i.path)))}
      title="Finish the selected drafts (the upscale passes, on the backend that rendered each); expired ones re-render first. Key U"
    >
      Upscale {ids.size} <span className="text-fg3">U</span>
    </Btn>
  );
}

/** Show / hide the render queue under the grid (count of jobs in it). */
function QueueToggle() {
  const open = useQueue((s) => s.open);
  const n = useHub((s) =>
    s.jobs.filter((j) => j.tool === 'texture_flow' && (j.status === 'queued' || j.status === 'running')).length
  );
  return (
    <Btn size="sm" tone="ghost" active={open} onClick={() => setQueueOpen(!open)} title="Show / hide the render queue (Modal + Local lanes)">
      Queue{n ? <span className="num text-deck">{n}</span> : null}
    </Btn>
  );
}

/** Videos in this folder the engine can't play yet and that have no HAP
 *  beside them (S8) — "Make playable" queues a `to_hap` job for each. */
function MakeFolderPlayable() {
  const listing = useStudio((s) => s.listing);
  const [busy, setBusy] = useState(false);
  const items = listing?.items ?? [];
  const hasHap = new Set(items.filter((i) => i.variant === 'hap' || i.codec === 'hap').map((i) => i.render_id));
  const todo = items.filter((i) => i.kind === 'video' && !i.playable && !hasHap.has(i.render_id));
  if (!todo.length) return null;
  return (
    <Btn
      size="sm"
      tone="ghost"
      disabled={busy}
      title="Write a HAP .mov beside every video here that has none, so it can go on a deck"
      onClick={async () => {
        setBusy(true);
        try {
          for (const it of todo) await startRender('to_hap', { video: it.path });
          useStore.getState().toast(`Making ${todo.length} file${todo.length === 1 ? '' : 's'} playable`, 'ok');
          refreshJobs();
        } catch (e) {
          useStore.getState().toast(`Make playable failed: ${e}`, 'bad');
        } finally {
          setBusy(false);
        }
      }}
    >
      Make playable ({todo.length})
    </Btn>
  );
}

const clampScore = (v: number) => Math.min(FULL_SCORES[1], Math.max(FULL_SCORES[0], v));

/** The score range filter: two thumbs over 0–9, integer steps. Press near a
 *  thumb to drag it (thumbs can't cross); double-click resets to the full
 *  range, which shows everything incl. unrated files. Looks like the app's
 *  RangeFader (ui/RangeFader.tsx), sized for a header. */
function ScoreFilter() {
  const range = useStudio((s) => s.scoreRange);
  const ref = useRef<HTMLDivElement | null>(null);
  const drag = useRef<'lo' | 'hi' | null>(null);
  const [lo, hi] = range;
  const steps = FULL_SCORES[1] - FULL_SCORES[0];
  const full = isFullRange(range);
  const valAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    return clampScore(Math.round(FULL_SCORES[0] + ((clientX - r.left) / r.width) * steps));
  };
  const apply = (which: 'lo' | 'hi', v: number) => {
    const next: ScoreRange = which === 'lo' ? [Math.min(v, hi), hi] : [lo, Math.max(v, lo)];
    if (next[0] !== lo || next[1] !== hi) useStudio.getState().setScoreRange(next);
  };
  const down = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const v = valAt(e.clientX);
    // The nearer thumb; on a tie (both thumbs on one value) the side pressed.
    const which = Math.abs(v - lo) < Math.abs(v - hi) ? 'lo' : Math.abs(v - lo) > Math.abs(v - hi) ? 'hi' : v < lo ? 'lo' : 'hi';
    drag.current = which;
    apply(which, v);
  };
  const move = (e: React.PointerEvent) => drag.current && apply(drag.current, valAt(e.clientX));
  const up = () => (drag.current = null);
  const pct = (v: number) => ((v - FULL_SCORES[0]) / steps) * 100;
  return (
    <div className="flex items-center gap-1.5 shrink-0" title="Show files scored in this range (double-click: all, incl. unrated)">
      <span className="text-[10px] text-fg3">score</span>
      <div
        ref={ref}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={up}
        onDoubleClick={() => useStudio.getState().setScoreRange(FULL_SCORES)}
        className="relative w-[110px] h-5 rounded-[5px] bg-panel2 border border-line hover:border-line2 touch-none select-none cursor-pointer"
      >
        {/* one tick per score */}
        {Array.from({ length: steps + 1 }, (_, i) => (
          <div key={i} className="absolute top-[7px] bottom-[7px] w-px bg-line2" style={{ left: `calc(${pct(i)}% - ${(pct(i) / 100) * 4 - 1.5}px)` }} />
        ))}
        <div
          className={'absolute top-0 bottom-0 ' + (full ? 'bg-fg3/10' : 'bg-deck/30')}
          style={{ left: `calc(${pct(lo)}% - ${pct(lo) * 0.04 - 2}px)`, right: `calc(${100 - pct(hi)}% + ${pct(hi) * 0.04 - 2}px)` }}
        />
        {[lo, hi].map((v, i) => (
          <div
            key={i}
            className={'absolute top-[2px] bottom-[2px] w-[4px] rounded-full ' + (full ? 'bg-fg3' : 'bg-deck')}
            // 0 sits flush left, 9 flush right: the 4 px thumb slides across the inner width.
            style={{ left: `calc(${pct(v)}% - ${(pct(v) / 100) * 4}px)` }}
          />
        ))}
      </div>
      <span className={'num text-[11px] w-7 ' + (full ? 'text-fg3' : 'text-deck')}>{full ? 'all' : lo === hi ? lo : `${lo}–${hi}`}</span>
    </div>
  );
}

/** "Delete N–M in this folder" (exactly what the range shows; unrated
 *  files never): dry run first, then a second click. */
function DeleteScored() {
  const range = useStudio((s) => s.scoreRange);
  const [armed, setArmed] = useState<{ range: ScoreRange; count: number } | null>(null);
  useEffect(() => setArmed(null), [range]);
  if (isFullRange(range)) return null;
  return armed ? (
    <Btn
      size="sm"
      tone="danger"
      onClick={async () => {
        await deleteScoredHere(armed.range, false);
        setArmed(null);
      }}
      title="Move the media files to the trash (restorable for 30 min); their records, labels and settings stay"
    >
      Delete {armed.count} file{armed.count === 1 ? '' : 's'}?
    </Btn>
  ) : (
    <Btn
      size="sm"
      tone="ghost"
      onClick={async () => {
        const r: any = await deleteScoredHere(range, true);
        const would: string[] = r?.would_delete ?? [];
        // A hub older than `min_score` would answer "≤ max": refuse rather than
        // delete files the range hides.
        const byPath = new Map((useStudio.getState().listing?.items ?? []).map((i) => [i.path, i.score]));
        if (would.some((p) => { const sc = byPath.get(p); return sc != null && sc < range[0]; })) {
          useStore.getState().toast('The content hub is out of date (ignores the range minimum) — restart it, then delete', 'bad');
          return;
        }
        setArmed({ range, count: would.length || (r?.count ?? 0) });
      }}
      title="Delete every file in this folder scored inside the range (unrated files are never touched)"
    >
      Delete {scoreRangeLabel(range)}…
    </Btn>
  );
}
