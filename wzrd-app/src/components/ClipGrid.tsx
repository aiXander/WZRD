// The deck grid (studio plan D1; was gig A2's clip grid). Resolume's clip
// stacks: deck tabs on top (one deck shown at a time — switching never
// changes what plays), an **incoming** strip of the newest playable renders
// made for this show, then one row per surface region (+ Unassigned) whose
// slots are the active deck's clips — library files referenced in place.
//
// A row is a radio group: a region plays at most one clip on DESIGN. Click
// an idle slot = play it there (state/clips.ts::bindClip → sceneCommit);
// click the playing slot again, or the row's ■ slot, = stop. TAKE as usual.
// DESIGN = design-colour ring, LIVE = red top bar. Drag a slot (or an
// incoming render) onto a row / slot to file it there; ✕ on hover removes
// it from the deck (the file stays). A slot whose file is gone shows ⚠ —
// "Relink missing" in the project menu finds it again (D3).

import { useEffect, useMemo, useRef, useState } from 'react';
import { libraryIncoming, libraryItems, mediaUrl, type LibraryItem } from '../api/hub';
import { bindClip, clipsByRegion, stopClip } from '../state/clips';
import {
  UNASSIGNED,
  absClip,
  activeDeck,
  addToDeck,
  createDeck,
  deleteDeck,
  moveDeck,
  moveSlot,
  removeSlot,
  renameDeck,
  setActive,
  useDecks,
  useDecksLivePoll,
  type Slot,
} from '../state/decks';
import { showName, useHub } from '../state/hub';
import { getMask, useMaskVersion } from '../state/masks';
import { hueForIndex, regionName } from '../state/scene';
import { useStore } from '../state/store';

const SLOT_DRAG = 'application/x-wzrd-slot';
const INCOMING_DRAG = 'application/x-wzrd-incoming';

type Row = { id: string; label: string; hue: number | null };
type SlotInfo = Partial<LibraryItem> & { path: string; exists?: boolean };

export function ClipGrid({ compact = false }: { compact?: boolean }) {
  const pack = useStore((s) => s.pack);
  const designJson = useStore((s) => s.sceneJson);
  const liveJson = useStore((s) => s.liveSceneJson);
  const doc = useDecks((s) => s.doc);
  const missing = useDecks((s) => s.missing);
  const error = useDecks((s) => s.error);
  useDecksLivePoll();
  useMaskVersion();
  const deck = activeDeck(doc);
  const info = useSlotInfo(doc);

  const design = useMemo(() => absMap(clipsByRegion(designJson)), [designJson]);
  const live = useMemo(() => absMap(clipsByRegion(liveJson)), [liveJson]);

  const rows: Row[] = useMemo(
    () => [
      ...(pack?.layers ?? []).map((l, i) => ({ id: l.id, label: regionName(pack, l.id), hue: hueForIndex(i) })),
      { id: UNASSIGNED, label: 'Unassigned', hue: null },
    ],
    [pack]
  );

  if (!pack) return <div className="p-3 text-[11px] text-fg3">Waiting for the layer pack…</div>;
  if (error && !doc) return <div className="p-3 text-[11px] text-bad">{error}</div>;
  if (!doc || !deck) return <div className="p-3 text-[11px] text-fg3">Loading decks…</div>;

  const inDeck = new Set(Object.values(deck.rows).flatMap((ss) => ss.filter(Boolean).map((s) => s!.path)));
  return (
    <div className="flex flex-col min-h-0 h-full">
      <DeckTabs compact={compact} />
      <Incoming compact={compact} inDeck={inDeck} />
      <div className="flex-1 min-h-0 overflow-auto">
        {inDeck.size === 0 && (
          <div className="px-3 py-2 text-[11px] text-fg3 leading-snug">
            This deck is empty. Add renders from Studio (“+ Deck”), drag one in from the incoming strip, or let the agent
            fill it (<span className="num">add_to_deck</span>).
          </div>
        )}
        {rows.map((row) => {
          const slots = deck.rows[row.id] ?? [];
          if (row.id === UNASSIGNED && !slots.some(Boolean)) return null;
          return (
            <RegionRow
              key={row.id}
              row={row}
              slots={slots}
              info={info}
              missing={missing}
              compact={compact}
              designClips={design[row.id] ?? []}
              liveClips={live[row.id] ?? []}
            />
          );
        })}
      </div>
    </div>
  );
}

function absMap(m: Record<string, string[]>): Record<string, string[]> {
  return Object.fromEntries(Object.entries(m).map(([k, v]) => [k, v.map(absClip)]));
}

/** Thumbnails + labels for every slot of every deck, one batch per change. */
function useSlotInfo(doc: ReturnType<typeof useDecks.getState>['doc']): Record<string, SlotInfo> {
  const [info, setInfo] = useState<Record<string, SlotInfo>>({});
  const key = useMemo(() => {
    const paths = new Set<string>();
    doc?.decks.forEach((d) => Object.values(d.rows).forEach((ss) => ss.forEach((s) => s && paths.add(s.path))));
    return [...paths].sort().join('\n');
  }, [doc]);
  useEffect(() => {
    if (!key) return setInfo({});
    let live = true;
    libraryItems(key.split('\n'))
      .then((items) => live && setInfo(Object.fromEntries(items.map((i) => [i.path, i]))))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [key]);
  return info;
}

function DeckTabs({ compact }: { compact: boolean }) {
  const doc = useDecks((s) => s.doc)!;
  const [editing, setEditing] = useState<string | null>(null);
  const [armed, setArmed] = useState<string | null>(null);
  const drag = useRef<string | null>(null);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(null), 2500);
    return () => clearTimeout(t);
  }, [armed]);
  return (
    <div className="flex items-center gap-1 px-2 py-1 border-b border-line shrink-0 overflow-x-auto">
      {doc.decks.map((d, i) => {
        const active = d.name === doc.active;
        return editing === d.name ? (
          <input
            key={d.name}
            autoFocus
            defaultValue={d.name}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === 'Enter') {
                renameDeck(d.name, (e.target as HTMLInputElement).value);
                setEditing(null);
              }
              if (e.key === 'Escape') setEditing(null);
            }}
            onBlur={(e) => {
              renameDeck(d.name, e.target.value);
              setEditing(null);
            }}
            className="h-6 w-24 px-1.5 text-[11px] rounded bg-bg border border-line2 outline-none"
          />
        ) : (
          <div
            key={d.name}
            draggable
            onDragStart={() => (drag.current = d.name)}
            onDragOver={(e) => drag.current && e.preventDefault()}
            onDrop={() => {
              if (drag.current && drag.current !== d.name) moveDeck(drag.current, i);
              drag.current = null;
            }}
            className={
              'group shrink-0 flex items-center h-6 rounded-md border text-[11px] ' +
              (active ? 'bg-raised border-line2 text-fg' : 'border-transparent text-fg3 hover:text-fg hover:bg-panel2')
            }
          >
            <button
              className="px-2 h-full"
              onClick={() => !active && setActive(d.name)}
              onDoubleClick={() => setEditing(d.name)}
              title={active ? 'Showing this deck — double-click to rename, drag to reorder' : 'Show this deck (what plays does not change)'}
            >
              {d.name}
            </button>
            {!compact && doc.decks.length > 1 && (
              <button
                className={
                  'pr-1.5 h-full text-[10px] ' +
                  (armed === d.name ? 'text-bad font-bold' : 'text-fg4 opacity-0 group-hover:opacity-100 hover:text-bad')
                }
                onClick={() => {
                  if (armed === d.name) {
                    deleteDeck(d.name);
                    setArmed(null);
                  } else setArmed(d.name);
                }}
                title={armed === d.name ? 'Click again to delete this deck (the clips stay in the library)' : 'Delete deck'}
              >
                {armed === d.name ? 'delete?' : '✕'}
              </button>
            )}
          </div>
        );
      })}
      <button
        onClick={() => createDeck()}
        className="shrink-0 h-6 w-6 rounded-md text-[13px] text-fg3 hover:text-fg hover:bg-panel2"
        title="New deck"
      >
        +
      </button>
    </div>
  );
}

/** The newest playable renders made for this show that aren't in the
 *  active deck yet. Click = add to the deck (in its region's row) and play
 *  it there on DESIGN; drag = file it on a row of your choice. */
function Incoming({ compact, inDeck }: { compact: boolean; inDeck: Set<string> }) {
  const up = useHub((s) => s.status?.state === 'up');
  const project = useStore((s) => s.project?.dir);
  const [items, setItems] = useState<LibraryItem[]>([]);
  useEffect(() => {
    const show = showName();
    if (!show) return;
    let live = true;
    const load = () =>
      libraryIncoming(show, 16)
        .then((r) => live && setItems(r))
        .catch(() => {});
    load();
    const t = window.setInterval(() => !document.hidden && load(), 4000);
    return () => {
      live = false;
      window.clearInterval(t);
    };
  }, [project, up]);
  const fresh = items.filter((i) => !inDeck.has(i.path));
  if (!fresh.length) return null;
  return (
    <div className="flex items-center gap-2 px-2 py-1 border-b border-amber/40 bg-amber/5 shrink-0">
      <div className={'shrink-0 text-[9px] font-bold tracking-[0.14em] text-amber ' + (compact ? 'w-[92px]' : 'w-[132px]')}>
        INCOMING · {fresh.length}
      </div>
      <div className="flex-1 min-w-0 flex gap-1.5 overflow-x-auto py-0.5">
        {fresh.map((it) => (
          <button
            key={it.path}
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData(INCOMING_DRAG, JSON.stringify(it));
              e.dataTransfer.effectAllowed = 'copy';
            }}
            onClick={async () => {
              if (!(await addToDeck([it]))) return;
              const row = it.region && useStore.getState().pack?.layers.some((l) => l.id === it.region) ? it.region : null;
              if (row) bindClip(it.path, row);
              else useStore.getState().toast('Added to Unassigned — drag it onto a region to play it', 'info');
            }}
            title={`${it.name}${it.region ? ` — made for ${it.region}` : ''}\nclick: add to this deck and play it on DESIGN · drag: file it on a row`}
            className={
              'group relative shrink-0 rounded-md overflow-hidden border-2 border-amber/60 bg-black ' +
              (compact ? 'w-[72px] h-[40px]' : 'w-[114px] h-[64px]')
            }
          >
            {it.thumb && <img src={mediaUrl(it.thumb)} draggable={false} className="absolute inset-0 w-full h-full object-cover" />}
            {!compact && (
              <span className="absolute inset-x-0 bottom-0 px-1 py-[1px] bg-black/65 text-[9px] text-fg2 truncate">{it.name}</span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function RegionRow({
  row,
  slots,
  info,
  missing,
  compact,
  designClips,
  liveClips,
}: {
  row: Row;
  slots: Slot[];
  info: Record<string, SlotInfo>;
  missing: Set<string>;
  compact: boolean;
  designClips: string[];
  liveClips: string[];
}) {
  const mask = row.id !== UNASSIGNED ? getMask(row.id) : undefined;
  const playable = row.id !== UNASSIGNED;
  const onDrop = (e: React.DragEvent, at?: number) => {
    const slot = e.dataTransfer.getData(SLOT_DRAG);
    const inc = e.dataTransfer.getData(INCOMING_DRAG);
    if (!slot && !inc) return;
    e.preventDefault();
    e.stopPropagation();
    if (slot) moveSlot(slot, row.id, at);
    else addToDeck([JSON.parse(inc)], row.id, at);
  };
  const accepts = (e: React.DragEvent) =>
    (e.dataTransfer.types.includes(SLOT_DRAG) || e.dataTransfer.types.includes(INCOMING_DRAG)) && e.preventDefault();
  return (
    <div className="flex items-stretch gap-2 px-2 py-1 border-b border-line/60 hover:bg-panel2/40" onDragOver={accepts} onDrop={(e) => onDrop(e)}>
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
        <span className={'truncate ' + (compact ? 'text-[10px]' : 'text-[11px]') + (playable ? ' text-fg' : ' text-fg3')}>{row.label}</span>
        <span className="num text-[9px] text-fg4 ml-auto">{slots.filter(Boolean).length || ''}</span>
      </div>
      <div className="flex-1 min-w-0 flex gap-1.5 overflow-x-auto py-0.5">
        {playable && (slots.some(Boolean) || designClips.length > 0 || liveClips.length > 0) ? (
          <StopSlot region={row.id} compact={compact} playing={designClips.length > 0} liveOnly={designClips.length === 0 && liveClips.length > 0} />
        ) : (
          <div className={'shrink-0 ' + (compact ? 'w-[30px]' : 'w-[44px]')} />
        )}
        {!slots.some(Boolean) && (
          <div className={'flex items-center text-[10px] text-fg4 ' + (compact ? 'h-[40px]' : 'h-[64px]')}>no clips in this deck — drop one here</div>
        )}
        {slots.map((s, i) =>
          s ? (
            <SlotTile
              key={s.path}
              slot={s}
              info={info[s.path]}
              region={playable ? row.id : null}
              compact={compact}
              missing={missing.has(s.path)}
              onDesign={designClips.includes(s.path)}
              onLive={liveClips.includes(s.path)}
              onDrop={(e) => onDrop(e, i)}
            />
          ) : (
            <div
              key={`empty-${i}`}
              onDragOver={accepts}
              onDrop={(e) => onDrop(e, i)}
              className={'shrink-0 rounded-md border border-dashed border-line ' + (compact ? 'w-[72px] h-[40px]' : 'w-[114px] h-[64px]')}
            />
          )
        )}
      </div>
    </div>
  );
}

function SlotTile({
  slot,
  info,
  region,
  compact,
  missing,
  onDesign,
  onLive,
  onDrop,
}: {
  slot: NonNullable<Slot>;
  info?: SlotInfo;
  region: string | null;
  compact: boolean;
  missing: boolean;
  onDesign: boolean;
  onLive: boolean;
  onDrop: (e: React.DragEvent) => void;
}) {
  const pack = useStore((s) => s.pack);
  const toast = useStore((s) => s.toast);
  const name = info?.name ?? slot.path.split('/').pop();
  const title = [
    name,
    [info?.tool, info?.created?.replace('T', ' ').slice(0, 16), info?.score != null ? `score ${info.score}` : null]
      .filter(Boolean)
      .join(' · ') || null,
    info?.region ? `made for ${regionName(pack, info.region)}` : null,
    missing ? 'FILE MISSING — project menu → Relink missing' : null,
    onDesign ? 'Playing on DESIGN — TAKE sends it to the wall' : onLive ? 'On the wall (LIVE)' : null,
    !region ? 'drag onto a region row to play it' : onDesign ? 'click: stop on DESIGN · drag: move' : 'click: play on this region in DESIGN · drag: move',
  ]
    .filter(Boolean)
    .join('\n');
  const tag = 'px-1 rounded-sm text-[8px] font-bold tracking-wider leading-[12px]';
  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(SLOT_DRAG, slot.path);
        e.dataTransfer.effectAllowed = 'move';
      }}
      onDragOver={(e) =>
        (e.dataTransfer.types.includes(SLOT_DRAG) || e.dataTransfer.types.includes(INCOMING_DRAG)) && e.preventDefault()
      }
      onDrop={onDrop}
      onClick={() => {
        if (!region) return toast('Drag this clip onto a region row first', 'warn');
        if (missing) return toast('This clip’s file is missing — project menu → Relink missing', 'warn');
        if (onDesign) stopClip(region, slot.path);
        else bindClip(slot.path, region);
      }}
      title={title}
      aria-pressed={onDesign}
      className={
        'group relative shrink-0 rounded-md overflow-hidden border-2 bg-black cursor-pointer ' +
        (compact ? 'w-[72px] h-[40px]' : 'w-[114px] h-[64px]') +
        ' ' +
        (missing
          ? 'border-amber/70 border-dashed'
          : onDesign
          ? 'border-design shadow-[0_0_10px_rgb(var(--design)/0.45)]'
          : 'border-line hover:border-line2')
      }
    >
      {info?.thumb && !missing ? (
        <img
          src={mediaUrl(info.thumb)}
          alt=""
          draggable={false}
          className={'absolute inset-0 w-full h-full object-cover ' + (onDesign || onLive ? '' : 'opacity-60 group-hover:opacity-90')}
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-[9px] text-fg4 bg-panel2 px-1 text-center">
          {missing ? '⚠ missing' : 'no preview'}
        </div>
      )}
      {onLive && <span className="absolute inset-x-0 top-0 h-[3px] bg-live" />}
      {(onDesign || onLive) && (
        <span className="absolute top-[4px] left-[3px] flex gap-[2px]">
          {onDesign && <span className={tag + ' bg-design text-black'}>DESIGN</span>}
          {onLive && <span className={tag + ' bg-live text-white'}>LIVE</span>}
        </span>
      )}
      {info?.score != null && !compact && (
        <span className="absolute top-[3px] right-[16px] num text-[9px] font-bold px-1 rounded-sm bg-black/70 text-amber">{info.score}</span>
      )}
      <button
        onClick={(e) => {
          e.stopPropagation();
          removeSlot(slot.path);
        }}
        className="absolute top-0 right-0 w-4 h-4 text-[10px] leading-none bg-black/70 text-fg3 hidden group-hover:block hover:text-bad"
        title="Remove from this deck (the file stays in the library)"
      >
        ✕
      </button>
      {!compact && <span className="absolute inset-x-0 bottom-0 px-1 py-[1px] bg-black/65 text-[9px] text-fg2 truncate">{name}</span>}
    </div>
  );
}

/** The row's "nothing" choice — the radio group's off position. */
function StopSlot({ region, compact, playing, liveOnly }: { region: string; compact: boolean; playing: boolean; liveOnly: boolean }) {
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
        (playing ? 'border-line text-fg3 hover:border-line2 hover:text-fg cursor-pointer' : 'border-line2 bg-panel2 text-fg2 cursor-default')
      }
    >
      <span>■</span>
      {!compact && <span className="text-[8px] font-bold tracking-wider">{playing ? 'STOP' : 'NONE'}</span>}
    </button>
  );
}
