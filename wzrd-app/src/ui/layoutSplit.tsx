// Resizable, rearrangeable room layout. Hand-rolled (no layout library):
// the rooms need px-sized side panels around one flexible centre, which is
// ~one flex rule plus a pointer drag — less than adapting a percentage-
// based split library to it.
//
//   <Split id="perform.cols" dir="row">
//     <Pane id="main" fill min={480} label="Stage">…</Pane>
//     <Pane id="side" size="316px" min={260} label="Deck">…</Pane>
//   </Split>
//
// - Exactly one `fill` pane per split takes the remaining space; every
//   other pane has a default `size` (any CSS length) until dragged, then a
//   stored px size (`ui/layoutStore.ts`, saved per project).
// - Dragging an edge resizes the pane on the far side of it from the fill
//   pane; the fill pane absorbs (and never goes below its `min`). Double-
//   click an edge = back to the default size.
// - Unlocked, each pane shows a ⠿ grip: drag it onto another pane of the
//   same split to swap them. Panes are keyed by id, so a swap moves DOM
//   nodes — nothing remounts (a native preview slot keeps its place in the
//   shell; `notifyLayout` makes it re-measure).
// - Locked (the default), edges are plain gaps and nothing reacts.
//
// The hole rule holds: Split and Pane wrappers paint no background, so a
// `<NativeSlot>` inside any pane still shows through.
//
// During a drag the size is written to the DOM directly (no React render
// per pointer move) and committed to the store on release.

import {
  Children,
  createContext,
  isValidElement,
  useContext,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
  type ReactNode,
} from 'react';
import { notifyLayout, setPaneOrder, setPaneSize, useLayout } from './layoutStore';

type PaneProps = {
  id: string;
  /** Default size along the split axis, any CSS length ('248px', '36%').
   *  Ignored for the `fill` pane. */
  size?: string;
  /** Minimum px along the split axis — the drag clamp and the CSS minimum. */
  min?: number;
  /** Takes the remaining space. Exactly one per split. */
  fill?: boolean;
  /** Name on the move grip while unlocked. */
  label?: string;
  className?: string;
  children?: ReactNode;
};

/** A slot in a `<Split>`; the Split renders it (this is a props marker). */
export function Pane(_: PaneProps): null {
  return null;
}

/** Nesting depth: grips of alternate depths sit at the top / bottom edge so
 *  an outer pane's grip never lands on its first inner pane's. */
const Depth = createContext(0);

export function Split({
  id,
  dir,
  className = '',
  rearrange = true,
  children,
}: {
  id: string;
  dir: 'row' | 'col';
  className?: string;
  /** Allow swapping panes by their grip (unlocked). */
  rearrange?: boolean;
  children: ReactNode;
}) {
  const depth = useContext(Depth);
  const locked = useLayout((s) => s.doc.locked);
  const sizes = useLayout((s) => s.doc.sizes);
  const saved = useLayout((s) => s.doc.order[id]);
  const els = useRef(new Map<string, HTMLDivElement>());
  const [drag, setDrag] = useState<{ from: string; over: string | null } | null>(null);
  const row = dir === 'row';

  const panes = Children.toArray(children).filter(
    (c): c is ReactElement<PaneProps> => isValidElement(c) && c.type === Pane
  );
  const byId = new Map(panes.map((p) => [p.props.id, p.props]));
  const order = [
    ...(saved ?? []).filter((x) => byId.has(x)),
    ...panes.map((p) => p.props.id).filter((x) => !(saved ?? []).includes(x)),
  ];
  // A pane appearing/disappearing (Prepare's drawer) or a swap can shift a
  // native slot without resizing it.
  const orderKey = order.join(',');
  useEffect(() => notifyLayout(), [orderKey]);
  let fillIdx = order.findIndex((x) => byId.get(x)!.fill);
  if (fillIdx < 0) fillIdx = order.length - 1;
  const canMove = !locked && rearrange && order.length > 1;

  function startResize(e: React.PointerEvent<HTMLDivElement>, i: number) {
    if (e.button !== 0) return;
    e.preventDefault();
    const tIdx = fillIdx <= i ? i + 1 : i;
    const grow = tIdx === i ? 1 : -1;
    const tId = order[tIdx];
    const tEl = els.current.get(tId);
    const fEl = els.current.get(order[fillIdx]);
    if (!tEl || !fEl) return;
    const along = (el: HTMLElement) => (row ? el.getBoundingClientRect().width : el.getBoundingClientRect().height);
    const start = along(tEl);
    const tMin = byId.get(tId)!.min ?? 40;
    const max = start + Math.max(0, along(fEl) - (byId.get(order[fillIdx])!.min ?? 80));
    const p0 = row ? e.clientX : e.clientY;
    const handle = e.currentTarget;
    handle.setPointerCapture(e.pointerId);
    document.body.style.cursor = row ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';
    let size = start;
    const move = (ev: PointerEvent) => {
      size = Math.min(max, Math.max(tMin, start + ((row ? ev.clientX : ev.clientY) - p0) * grow));
      tEl.style.flexBasis = `${size}px`;
      notifyLayout();
    };
    const up = () => {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.removeEventListener('pointercancel', up);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      if (Math.abs(size - start) >= 1) setPaneSize(`${id}/${tId}`, size);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
    handle.addEventListener('pointercancel', up);
  }

  function resetEdge(i: number) {
    const tId = order[fillIdx <= i ? i + 1 : i];
    setPaneSize(`${id}/${tId}`, null);
  }

  function startMove(e: React.PointerEvent<HTMLButtonElement>, from: string) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const grip = e.currentTarget;
    grip.setPointerCapture(e.pointerId);
    document.body.style.cursor = 'grabbing';
    let over: string | null = null;
    setDrag({ from, over: null });
    const sel = `[data-split="${CSS.escape(id)}"]`;
    const move = (ev: PointerEvent) => {
      const hit = document.elementFromPoint(ev.clientX, ev.clientY)?.closest<HTMLElement>(sel);
      const next = hit?.dataset.pane && hit.dataset.pane !== from ? hit.dataset.pane : null;
      if (next !== over) {
        over = next;
        setDrag({ from, over });
      }
    };
    const up = () => {
      grip.removeEventListener('pointermove', move);
      grip.removeEventListener('pointerup', up);
      grip.removeEventListener('pointercancel', up);
      document.body.style.cursor = '';
      setDrag(null);
      if (!over) return;
      const next = [...order];
      const a = next.indexOf(from);
      const b = next.indexOf(over);
      [next[a], next[b]] = [next[b], next[a]];
      setPaneOrder(id, next);
    };
    grip.addEventListener('pointermove', move);
    grip.addEventListener('pointerup', up);
    grip.addEventListener('pointercancel', up);
  }

  const out: ReactNode[] = [];
  order.forEach((pid, i) => {
    const p = byId.get(pid)!;
    if (i > 0) {
      const edge = i - 1;
      out.push(
        locked ? (
          <div key={`edge:${pid}`} className={row ? 'w-2 shrink-0' : 'h-2 shrink-0'} />
        ) : (
          <div
            key={`edge:${pid}`}
            onPointerDown={(e) => startResize(e, edge)}
            onDoubleClick={() => resetEdge(edge)}
            title="Drag to resize · double-click: default size"
            className={
              'group relative shrink-0 flex items-center justify-center touch-none ' +
              (row ? 'w-2 cursor-col-resize' : 'h-2 cursor-row-resize')
            }
          >
            <div
              className={
                'absolute rounded-full bg-deck opacity-0 group-hover:opacity-70 group-active:opacity-100 ' +
                (row ? 'inset-y-1 w-[2px]' : 'inset-x-1 h-[2px]')
              }
            />
            <div className={'rounded-full bg-fg3/70 ' + (row ? 'w-[2px] h-6' : 'h-[2px] w-6')} />
          </div>
        )
      );
    }
    const basis = p.fill ? '0px' : sizes[`${id}/${pid}`] != null ? `${sizes[`${id}/${pid}`]}px` : p.size ?? 'auto';
    const style: CSSProperties = { flex: `${p.fill ? 1 : 0} 1 ${basis}` };
    if (p.min != null) style[row ? 'minWidth' : 'minHeight'] = p.min;
    const isOver = drag?.over === pid;
    const isFrom = drag?.from === pid;
    out.push(
      <div
        key={pid}
        ref={(el) => {
          if (el) els.current.set(pid, el);
          else els.current.delete(pid);
        }}
        data-split={id}
        data-pane={pid}
        style={style}
        className={
          'relative flex flex-col min-w-0 min-h-0 [&>*]:flex-1 [&>*]:min-h-0 [&>*]:min-w-0 rounded-lg ' +
          (isOver
            ? 'outline outline-2 outline-dashed outline-deck '
            : isFrom
            ? 'outline outline-2 outline-dotted outline-fg3 '
            : '') +
          (p.className ?? '')
        }
      >
        {p.children}
        {canMove && (
          <button
            type="button"
            onPointerDown={(e) => startMove(e, pid)}
            title={`Drag onto another panel to swap places`}
            className={
              'absolute left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 h-[18px] px-2 rounded-full ' +
              'bg-raised/95 border border-line2 text-[10px] text-fg2 hover:text-fg hover:border-deck/70 cursor-grab touch-none whitespace-nowrap ' +
              (depth % 2 === 0 ? 'top-1' : 'bottom-1')
            }
          >
            <span className="text-fg3">⠿</span>
            {p.label ?? pid}
          </button>
        )}
      </div>
    );
  });

  return (
    <Depth.Provider value={depth + 1}>
      <div className={'flex min-w-0 min-h-0 ' + (row ? 'flex-row ' : 'flex-col ') + className}>{out}</div>
    </Depth.Provider>
  );
}
