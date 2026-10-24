// Drag library paths between panels with pointer events (Studio grid →
// style tray, Render panel slots, …).
//
// Why not HTML5 drag-and-drop: the shell keeps Tauri's native drag-drop on
// (Finder drops into show setup need it — `useFileDrop.ts`), and on macOS
// that native handler swallows in-page HTML5 drags: `drop` never reaches the
// page. It also can't be scripted from CDP. So a drag here is plain pointer
// events: press on a source, move past a 5 px threshold, a small ghost
// (thumb + count) follows the cursor, the target under it (hit-tested via
// `[data-path-drop]`) lights up, release drops. A release elsewhere does
// nothing; the click that a drag would otherwise produce is swallowed.
//
//   source: onPointerDown={(e) => startPathDrag(e, () => ({ paths, thumb }))}
//   target: const { dropProps, hovering } = usePathDrop((paths) => …, accept?)
//           <div {...dropProps} className={hovering ? 'ring-1 ring-deck' : ''}>

import { useEffect, useRef, useState } from 'react';
import { mediaUrl } from '../../api/hub';

type Target = {
  accept: (paths: string[]) => string[];
  onDrop: (paths: string[]) => void;
  setHover: (h: boolean) => void;
};

const targets = new Map<string, Target>();
let seq = 0;
const THRESHOLD = 5;

/** A drop zone for library paths. `accept` filters what it takes (nothing left = no hover, no drop). */
export function usePathDrop(onDrop: (paths: string[]) => void, accept: (paths: string[]) => string[] = (p) => p) {
  const [hovering, setHover] = useState(false);
  const id = useRef(`pd${++seq}`).current;
  const t = useRef<Target>({ accept, onDrop, setHover });
  t.current.accept = accept;
  t.current.onDrop = onDrop;
  useEffect(() => {
    targets.set(id, t.current);
    return () => {
      targets.delete(id);
    };
  }, [id]);
  return { dropProps: { 'data-path-drop': id }, hovering };
}

function targetAt(x: number, y: number): Target | null {
  const el = document.elementFromPoint(x, y)?.closest<HTMLElement>('[data-path-drop]');
  return (el && targets.get(el.dataset.pathDrop ?? '')) || null;
}

function ghost(paths: string[], thumb: string | null): HTMLDivElement {
  const g = document.createElement('div');
  g.style.cssText =
    'position:fixed;left:0;top:0;z-index:100;pointer-events:none;width:56px;height:56px;border-radius:6px;' +
    'background:#000 center/cover no-repeat;border:1px solid rgb(var(--deck));box-shadow:0 8px 24px rgba(0,0,0,.6);opacity:.9';
  if (thumb) g.style.backgroundImage = `url("${mediaUrl(thumb)}")`;
  if (paths.length > 1) {
    const n = document.createElement('span');
    n.textContent = String(paths.length);
    n.style.cssText =
      'position:absolute;right:-6px;top:-6px;min-width:18px;height:18px;padding:0 4px;border-radius:9px;' +
      'background:rgb(var(--deck));color:#000;font:600 11px/18px -apple-system,sans-serif;text-align:center';
    g.appendChild(n);
  }
  document.body.appendChild(g);
  return g;
}

/** Start a (potential) drag from a pointerdown. `payload` is read once the
 *  threshold is crossed, so a plain click never pays for it. */
export function startPathDrag(e: React.PointerEvent, payload: () => { paths: string[]; thumb: string | null }) {
  if (e.button !== 0 || e.shiftKey || e.metaKey || e.ctrlKey) return;
  const x0 = e.clientX;
  const y0 = e.clientY;
  let drag: { paths: string[]; g: HTMLDivElement } | null = null;
  let over: Target | null = null;
  const hover = (t: Target | null) => {
    if (t === over) return;
    over?.setHover(false);
    over = t;
    over?.setHover(true);
  };
  const move = (ev: PointerEvent) => {
    if (!drag) {
      if (Math.hypot(ev.clientX - x0, ev.clientY - y0) < THRESHOLD) return;
      const p = payload();
      if (!p.paths.length) return end();
      drag = { paths: p.paths, g: ghost(p.paths, p.thumb) };
      document.body.style.cursor = 'grabbing';
    }
    // Above-right of the cursor: drop zones sit low (the tray), a ghost below would clip.
    drag.g.style.transform = `translate(${ev.clientX + 12}px, ${ev.clientY - 68}px)`;
    const t = targetAt(ev.clientX, ev.clientY);
    hover(t && t.accept(drag.paths).length ? t : null);
  };
  const up = () => {
    const t = over;
    const d = drag;
    end();
    if (!d) return;
    // The click a drag ends with must not select / open anything.
    const swallow = (ev: MouseEvent) => ev.stopPropagation();
    window.addEventListener('click', swallow, { capture: true, once: true });
    setTimeout(() => window.removeEventListener('click', swallow, { capture: true }), 0);
    if (t) {
      const took = t.accept(d.paths);
      if (took.length) t.onDrop(took);
    }
  };
  function end() {
    hover(null);
    drag?.g.remove();
    drag = null;
    document.body.style.cursor = '';
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    window.removeEventListener('pointercancel', end);
  }
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
  window.addEventListener('pointercancel', end);
}
