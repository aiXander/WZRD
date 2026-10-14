// A2 — the region × clip-slot grid (gig plan D12). Resolume's layer rows,
// but each row is a named surface region: the clips rendered *for* that
// region sit in its row, newest first; clips made for no region sit in
// "Unassigned". Drag a clip onto another row to re-file it.
//
// A row is a radio group: a region plays at most one grid clip on DESIGN.
// Click an idle slot = play it there, replacing the row's current clip
// (state/clips.ts::bindClip → sceneCommit); click the playing slot again, or
// the row's leading ■ slot, = stop (clips.ts::stopClip). TAKE as usual.
// Three looks, never confused: DESIGN (design-colour ring + DESIGN tag —
// what TAKE would send), LIVE (red top bar + LIVE tag — on the wall now),
// idle (no ring, dimmed). Hover says what a click will do. Fresh agent
// renders are flagged NEW and also lined up in the "new from agent" lane on
// top, so the operator never hunts for them.

import { useMemo } from 'react';
import type { ClipEntry } from '../api/ipc';
import {
  bindClip,
  clipsByRegion,
  fileClip,
  rowOf,
  stopClip,
  useClips,
  useClipsLivePoll,
  useThumb,
} from '../state/clips';
import { getMask, useMaskVersion } from '../state/masks';
import { hueForIndex, regionName } from '../state/scene';
import { useStore } from '../state/store';

const DRAG_TYPE = 'text/x-wzrd-clip';

type Row = { id: string | null; label: string; hue: number | null };

export function ClipGrid({ compact = false }: { compact?: boolean }) {
  const pack = useStore((s) => s.pack);
  const designJson = useStore((s) => s.sceneJson);
  const liveJson = useStore((s) => s.liveSceneJson);
  const clips = useClips((s) => s.clips);
  const rows = useClips((s) => s.rows);
  const seen = useClips((s) => s.seen);
  useClipsLivePoll();
  useMaskVersion();

  const design = useMemo(() => clipsByRegion(designJson), [designJson]);
  const live = useMemo(() => clipsByRegion(liveJson), [liveJson]);

  const rowList: Row[] = useMemo(
    () => [
      ...(pack?.layers ?? []).map((l, i) => ({ id: l.id, label: regionName(pack, l.id), hue: hueForIndex(i) })),
      { id: null, label: 'Unassigned', hue: null },
    ],
    [pack]
  );
  const byRow = useMemo(() => {
    const m = new Map<string | null, ClipEntry[]>();
    for (const c of clips) {
      const r = rowOf(c, rows, pack);
      m.set(r, [...(m.get(r) ?? []), c]);
    }
    return m;
  }, [clips, rows, pack]);
  const fresh = clips.filter((c) => !seen.has(c.path));

  if (!pack) return <div className="p-3 text-[11px] text-fg3">Waiting for the layer pack…</div>;

  return (
    <div className="flex flex-col min-h-0 h-full">
      <NewLane
        clips={fresh}
        compact={compact}
        rowFor={(c) => rowOf(c, rows, pack)}
        design={design}
        live={live}
      />
      <div className="flex-1 min-h-0 overflow-auto">
        {clips.length === 0 && (
          <div className="px-3 py-2 text-[11px] text-fg3 leading-snug">
            No clips in this project yet. Agent renders (HAP <span className="num">.mov</span> in{' '}
            <span className="num">generated/</span>) appear here on their own, in the row of the region they were made for.
          </div>
        )}
        {rowList.map((row) => {
          const items = byRow.get(row.id) ?? [];
          if (row.id === null && items.length === 0) return null;
          return (
            <RegionRow
              key={row.id ?? '__unassigned'}
              row={row}
              clips={items}
              compact={compact}
              designClips={row.id ? design[row.id] ?? [] : []}
              liveClips={row.id ? live[row.id] ?? [] : []}
            />
          );
        })}
      </div>
    </div>
  );
}

function RegionRow({
  row,
  clips,
  compact,
  designClips,
  liveClips,
}: {
  row: Row;
  clips: ClipEntry[];
  compact: boolean;
  designClips: string[];
  liveClips: string[];
}) {
  const mask = row.id ? getMask(row.id) : undefined;
  return (
    <div
      className="flex items-stretch gap-2 px-2 py-1 border-b border-line/60 hover:bg-panel2/40"
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes(DRAG_TYPE)) e.preventDefault();
      }}
      onDrop={(e) => {
        const path = e.dataTransfer.getData(DRAG_TYPE);
        if (path) fileClip(path, row.id);
      }}
    >
      <div className={'shrink-0 flex items-center gap-1.5 ' + (compact ? 'w-[92px]' : 'w-[132px]')}>
        {row.hue != null ? (
          <span
            className="w-4 h-3 shrink-0 rounded-[2px]"
            style={{
              background: `hsl(${row.hue} 85% 62%)`,
              WebkitMaskImage: mask ? `url(${mask.thumb})` : undefined,
              WebkitMaskSize: '100% 100%',
            }}
          />
        ) : (
          <span className="w-4 h-3 shrink-0 rounded-[2px] border border-dashed border-fg4" />
        )}
        <span className={'truncate ' + (compact ? 'text-[10px]' : 'text-[11px]') + (row.id ? ' text-fg' : ' text-fg3')}>
          {row.label}
        </span>
        <span className="num text-[9px] text-fg4 ml-auto">{clips.length || ''}</span>
      </div>
      <div className="flex-1 min-w-0 flex gap-1.5 overflow-x-auto py-0.5">
        {row.id && (clips.length > 0 || designClips.length > 0 || liveClips.length > 0) ? (
          <StopSlot region={row.id} compact={compact} playing={designClips.length > 0} liveOnly={designClips.length === 0 && liveClips.length > 0} />
        ) : (
          <div className={'shrink-0 ' + (compact ? 'w-[30px]' : 'w-[44px]')} />
        )}
        {clips.length === 0 && (
          <div className={'flex items-center text-[10px] text-fg4 ' + (compact ? 'h-[40px]' : 'h-[64px]')}>
            no clips for this region — drop one here
          </div>
        )}
        {clips.map((c) => (
          <Slot
            key={c.path}
            clip={c}
            region={row.id}
            compact={compact}
            onDesign={designClips.includes(c.path)}
            onLive={liveClips.includes(c.path)}
          />
        ))}
      </div>
    </div>
  );
}

function Slot({
  clip,
  region,
  compact,
  onDesign,
  onLive,
  showRegion = false,
}: {
  clip: ClipEntry;
  region: string | null;
  compact: boolean;
  onDesign: boolean;
  onLive: boolean;
  showRegion?: boolean;
}) {
  const thumb = useThumb(clip);
  const fresh = useClips((s) => !s.seen.has(clip.path));
  const pack = useStore((s) => s.pack);
  const toast = useStore((s) => s.toast);
  const sc = clip.sidecar;
  const origin = clip.source ?? sc;
  const prompt = origin?.args?.prompt;
  const madeFor = sc?.region ? regionName(pack, sc.region) : null;
  const state = onDesign
    ? 'Playing on DESIGN (the preview) — TAKE sends it to the wall' + (onLive ? '; it is on the wall already' : '')
    : onLive
      ? 'On the wall (LIVE) — DESIGN has moved on; TAKE replaces it'
      : null;
  const action = !region
    ? 'drag onto a region row to play it'
    : onDesign
      ? 'click: stop it on DESIGN · drag: re-file'
      : 'click: play on this region in DESIGN (replaces the clip it plays now) · drag: re-file';
  const title = [
    clip.name,
    prompt ? `“${String(prompt).slice(0, 160)}”` : null,
    [origin?.tool, origin?.created?.replace('T', ' ').slice(0, 16), origin?.est_usd ? `$${origin.est_usd.toFixed(2)}` : null]
      .filter(Boolean)
      .join(' · ') || null,
    madeFor ? `made for ${madeFor}` : 'made for no region',
    state,
    action,
  ]
    .filter(Boolean)
    .join('\n');
  const tag = 'px-1 rounded-sm text-[8px] font-bold tracking-wider leading-[12px]';

  return (
    <button
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(DRAG_TYPE, clip.path);
        e.dataTransfer.effectAllowed = 'move';
      }}
      onClick={() => {
        if (!region) {
          toast('This clip has no region yet — drag it onto a region row first', 'warn');
          return;
        }
        if (onDesign) stopClip(region, clip.path);
        else bindClip(clip.path, region);
      }}
      title={title}
      aria-pressed={onDesign}
      className={
        'group relative shrink-0 rounded-md overflow-hidden border-2 bg-black text-left ' +
        (compact ? 'w-[72px] h-[40px]' : 'w-[114px] h-[64px]') +
        ' ' +
        (onDesign ? 'border-design shadow-[0_0_10px_rgb(var(--design)/0.45)]' : 'border-line hover:border-line2')
      }
    >
      {thumb ? (
        <img
          src={thumb}
          alt=""
          draggable={false}
          className={
            'absolute inset-0 w-full h-full object-cover ' +
            (onDesign || onLive ? '' : 'opacity-60 group-hover:opacity-90')
          }
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-[9px] text-fg4 bg-panel2">
          {clip.thumb ? '…' : 'no preview'}
        </div>
      )}
      {onLive && <span className="absolute inset-x-0 top-0 h-[3px] bg-live" />}
      {(onDesign || onLive) && (
        <span className="absolute top-[4px] left-[3px] flex gap-[2px]">
          {onDesign && <span className={tag + ' bg-design text-black'}>DESIGN</span>}
          {onLive && <span className={tag + ' bg-live text-white'}>LIVE</span>}
        </span>
      )}
      {region && (
        <span className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity text-[10px] font-semibold tracking-wider text-fg pointer-events-none">
          {onDesign ? '■ STOP' : '▶ PLAY'}
        </span>
      )}
      {fresh && (
        <span className="absolute top-[3px] right-[3px] px-1 rounded-sm bg-amber text-black text-[8px] font-bold tracking-wider">
          NEW
        </span>
      )}
      {!compact && (
        <span className="absolute inset-x-0 bottom-0 px-1 py-[1px] bg-black/65 text-[9px] text-fg2 truncate">
          {showRegion && region ? `${regionName(pack, region)} · ` : ''}
          {clip.name}
        </span>
      )}
    </button>
  );
}

/** The "new from agent" lane: every NEW clip, newest first, one click to
 *  play it on the region it was made for. Hidden when nothing is new. */
function NewLane({
  clips,
  compact,
  rowFor,
  design,
  live,
}: {
  clips: ClipEntry[];
  compact: boolean;
  rowFor: (c: ClipEntry) => string | null;
  design: Record<string, string[]>;
  live: Record<string, string[]>;
}) {
  if (clips.length === 0) return null;
  return (
    <div className="flex items-center gap-2 px-2 py-1 border-b border-amber/40 bg-amber/5 shrink-0">
      <div className={'shrink-0 text-[9px] font-bold tracking-[0.14em] text-amber ' + (compact ? 'w-[92px]' : 'w-[132px]')}>
        NEW FROM AGENT · {clips.length}
      </div>
      <div className="flex-1 min-w-0 flex gap-1.5 overflow-x-auto py-0.5">
        {clips.map((c) => {
          const r = rowFor(c);
          return (
            <Slot
              key={c.path}
              clip={c}
              region={r}
              compact={compact}
              onDesign={!!r && (design[r] ?? []).includes(c.path)}
              onLive={!!r && (live[r] ?? []).includes(c.path)}
              showRegion
            />
          );
        })}
      </div>
    </div>
  );
}

/** The row's "nothing" choice — the radio group's off position. Lit while
 *  the region plays no grid clip on DESIGN; a click stops whatever it plays. */
function StopSlot({
  region,
  compact,
  playing,
  liveOnly,
}: {
  region: string;
  compact: boolean;
  playing: boolean;
  liveOnly: boolean;
}) {
  return (
    <button
      onClick={() => playing && stopClip(region)}
      aria-pressed={!playing}
      title={
        playing
          ? 'Stop this region’s clip on DESIGN (TAKE to stop it on the wall too)'
          : liveOnly
            ? 'Nothing plays here on DESIGN — the wall still shows a clip until TAKE'
            : 'Nothing plays on this region — click a clip to play it'
      }
      className={
        'shrink-0 flex flex-col items-center justify-center gap-0.5 rounded-md border-2 ' +
        (compact ? 'w-[30px] h-[40px] text-[11px]' : 'w-[44px] h-[64px] text-[13px]') +
        ' ' +
        (playing
          ? 'border-line text-fg3 hover:border-line2 hover:text-fg cursor-pointer'
          : 'border-line2 bg-panel2 text-fg2 cursor-default')
      }
    >
      <span>■</span>
      {!compact && <span className="text-[8px] font-bold tracking-wider">{playing ? 'STOP' : 'NONE'}</span>}
    </button>
  );
}
