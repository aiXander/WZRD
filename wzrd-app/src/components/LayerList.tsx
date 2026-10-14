// Prepare's left column: the layer list (one row per design binding) and
// the region list (one row per pack region). Selection rules shared with the
// surface map: clicking the selected thing again deselects it, Esc clears
// both selections (shortcuts.ts). Each layer row has a delete × (always shown
// on the selected row, on hover otherwise); deletes are undoable (⌘Z or the
// header's Undo — actions.ts::deleteLayer).

import { useSyncExternalStore } from 'react';
import { deleteLayer, lastDeletedLayer, subscribeLayerUndo, undoDeleteLayer } from '../state/actions';
import { getMask, useMaskVersion } from '../state/masks';
import {
  bindingHue,
  effectName,
  hueForIndex,
  layerTitle,
  prettify,
  regionName,
  selectorCovers,
  targetLabel,
  type Binding,
} from '../state/scene';
import { useStore } from '../state/store';
import { Btn, Panel } from '../ui/primitives';
import { RegionGlyph } from './LayerStack';

export function LayerList({
  bindings,
  armed,
  onNew,
  className = '',
}: {
  bindings: Binding[];
  /** "+ New" is armed: the next region click creates a layer. */
  armed: boolean;
  onNew: () => void;
  className?: string;
}) {
  const pack = useStore((s) => s.pack);
  const selected = useStore((s) => s.selectedBindingId);
  const setSelected = useStore((s) => s.setSelectedBindingId);
  const setRegion = useStore((s) => s.setSelectedLayerId);
  const setHovered = useStore((s) => s.setHoveredBindingId);
  const undoable = useSyncExternalStore(subscribeLayerUndo, lastDeletedLayer);

  function pick(id: string) {
    // A layer row aims the inspector; a stale region pick would keep glowing
    // on the map, so it goes. Clicking the selected row again deselects.
    setRegion(null);
    setSelected(id === selected ? null : id);
  }

  return (
    <Panel
      title="Layers"
      right={
        <div className="flex items-center gap-1">
          {undoable && (
            <Btn size="sm" tone="ghost" onClick={() => undoDeleteLayer()} title={`Restore “${prettify(undoable)}” (⌘Z)`}>
              ↶ Undo
            </Btn>
          )}
          <Btn size="sm" tone="design" active={armed} onClick={onNew} title="New layer on the selected region — or click a region on the map next">
            + New
          </Btn>
        </div>
      }
      className={className}
      bodyClassName="overflow-auto"
    >
      <div className="flex flex-col p-1.5 gap-1" onMouseLeave={() => setHovered(null)}>
        {bindings.length === 0 && (
          <div className="p-2 text-[11px] text-fg3 leading-snug">
            The surface is dark. Click a region on the map, then <b className="text-fg2">+ New</b>.
          </div>
        )}
        {bindings.map((b, i) => {
          const sel = b.id === selected;
          return (
            <div
              key={b.id}
              role="button"
              tabIndex={-1}
              onMouseEnter={() => setHovered(b.id)}
              onClick={() => pick(b.id)}
              title={sel ? 'Click again to deselect (Esc) · Delete / ⌫ removes it' : undefined}
              className={
                'group flex items-center gap-2 h-10 pl-1.5 pr-1 rounded-md border text-left cursor-pointer ' +
                (sel ? 'bg-raised border-design/60' : 'border-transparent hover:bg-panel2')
              }
            >
              <RegionGlyph binding={b} pack={pack} hue={bindingHue(b, pack, i)} size={26} />
              <div className="min-w-0 flex-1">
                <div className="text-[12px] font-medium truncate">{layerTitle(b)}</div>
                <div className="text-[10px] text-fg3 truncate">
                  {targetLabel(b, pack)} · {prettify(effectName(b.effect))}
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteLayer(b.id);
                }}
                title="Delete layer (Delete / ⌫) — ⌘Z restores it"
                aria-label={`Delete ${layerTitle(b)}`}
                className={
                  'w-6 h-6 shrink-0 rounded flex items-center justify-center text-[14px] leading-none text-fg3 ' +
                  'hover:text-bad hover:bg-bad/15 focus:opacity-100 ' +
                  (sel ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')
                }
              >
                ×
              </button>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

export function RegionList({
  bindings,
  onPick,
  onCreate,
}: {
  bindings: Binding[];
  onPick: (id: string) => void;
  onCreate: (id: string) => void;
}) {
  const pack = useStore((s) => s.pack);
  const sel = useStore((s) => s.selectedLayerId);
  useMaskVersion();
  if (!pack) return null;
  return (
    <div className="flex flex-col p-1.5 gap-0.5">
      {pack.layers.map((l, i) => {
        const n = bindings.filter((b) => selectorCovers(b.select, pack, l)).length;
        const m = getMask(l.id);
        const hue = hueForIndex(i);
        return (
          <div
            key={l.id}
            onClick={() => onPick(l.id)}
            onDoubleClick={() => onCreate(l.id)}
            title={`${l.id}${l.tags.length ? ' · #' + l.tags.join(' #') : ''}\nclick: select (again: next layer / deselect) · double-click: new layer here`}
            className={
              'flex items-center gap-2 h-7 px-1.5 rounded-md cursor-pointer ' +
              (sel === l.id ? 'bg-raised ring-1 ring-design/50' : 'hover:bg-panel2')
            }
          >
            <span
              className="w-4 h-3 shrink-0 rounded-[2px]"
              style={{
                background: `hsl(${hue} 85% 62%)`,
                WebkitMaskImage: m ? `url(${m.thumb})` : undefined,
                WebkitMaskSize: '100% 100%',
              }}
            />
            <span className="text-[11px] truncate flex-1">{regionName(pack, l.id)}</span>
            {l.tags.slice(0, 1).map((t) => (
              <span key={t} className="text-[9px] text-fg4">
                #{t}
              </span>
            ))}
            <span className={'num text-[10px] w-4 text-right ' + (n ? 'text-design' : 'text-fg4')}>{n || '·'}</span>
          </div>
        );
      })}
    </div>
  );
}
