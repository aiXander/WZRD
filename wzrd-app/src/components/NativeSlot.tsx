// A hole in the page the native GPU preview shows through
// (docs/TODO/native-preview-underlay.md §4.3). The slot measures its own
// rect and pushes it, with what to show (`params`), to the shell, which
// places the engine's preview view exactly there — *beneath* the
// transparent webview, so `children` (mask outlines, the warp grid,
// labels, dropdowns) draw over the image.
//
// The hole rule: nothing between the slot and the page root may paint a
// background over the slot's rect, or it silently hides the image. A
// dev-only check walks the ancestors on mount and outlines offenders red.
//
// Outside the underlay (plain browser, or `WZRD_PREVIEW=window` for every
// slot but Perform's) the caller renders its own JPEG fallback as
// `children`; the slot then pushes nothing.

import { useEffect, useRef, type CSSProperties, type ReactNode } from 'react';
import { previewSetSlot, type PreviewParams } from '../api/ipc';
import { previewMode } from '../api/previewMode';
import { LAYOUT_EVENT } from '../ui/layoutStore';

export function NativeSlot({
  id = 'main',
  params,
  native = previewMode() === 'underlay',
  className = '',
  style,
  children,
}: {
  /** `studio` = the Studio monitor (shows the `previewFile` source). */
  id?: 'main' | 'studio';
  params?: PreviewParams;
  /** Push this slot to the shell. Defaults to "the underlay is active";
   *  Perform also passes true in `window` mode (the child window). */
  native?: boolean;
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Params are a small literal per render; key the effect on their content.
  const paramsKey = JSON.stringify(params ?? {});

  useEffect(() => {
    if (!native) return;
    const el = ref.current;
    if (!el) return;
    const p: PreviewParams = JSON.parse(paramsKey);
    let raf = 0;
    // Last rect sent: the capture-phase scroll listener fires for *any*
    // scrolling element on the page (a layer list, the log), and each push
    // is a main-thread Tauri command — only send when the slot moved.
    let last = '';
    let forcePending = false;
    const push = (force = false) => {
      forcePending ||= force;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const force = forcePending;
        forcePending = false;
        const r = el.getBoundingClientRect();
        const slot = {
          x: r.left,
          y: r.top,
          width: r.width,
          height: r.height,
          visible: r.width > 4 && r.height > 4,
        };
        const key = `${slot.x},${slot.y},${slot.width},${slot.height},${slot.visible}`;
        if (!force && key === last) return;
        last = key;
        previewSetSlot({ id, ...slot, params: p }).catch(() => {
          last = ''; // retry on the next event
        });
      });
    };
    const onMove = () => push();
    // Always send on window resize (the backend maps CSS px through the
    // window's current geometry) and focus (the `window`-mode child window
    // detaches after minimize / display changes; every push re-attaches).
    const onForce = () => push(true);
    push(true);
    const ro = new ResizeObserver(onMove);
    ro.observe(el);
    window.addEventListener('resize', onForce);
    window.addEventListener('scroll', onMove, true);
    // A panel resize/swap (ui/layoutSplit.tsx) can move the slot without
    // resizing it, which the ResizeObserver doesn't see.
    window.addEventListener(LAYOUT_EVENT, onMove);
    if (previewMode() === 'window') window.addEventListener('focus', onForce);
    if (import.meta.env.DEV && previewMode() === 'underlay') checkHole(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', onForce);
      window.removeEventListener('scroll', onMove, true);
      window.removeEventListener(LAYOUT_EVENT, onMove);
      window.removeEventListener('focus', onForce);
      previewSetSlot({ x: 0, y: 0, width: 0, height: 0, visible: false }).catch(() => {});
    };
  }, [native, paramsKey, id]);

  return (
    <div ref={ref} className={className} style={style}>
      {children}
    </div>
  );
}

/** Dev-only: flag every ancestor (and the slot itself) that paints a
 *  background — each one hides the native image under it. */
function checkHole(slot: HTMLElement) {
  for (let el: HTMLElement | null = slot; el; el = el.parentElement) {
    const cs = getComputedStyle(el);
    const alpha = /rgba?\(([^)]+)\)/.exec(cs.backgroundColor)?.[1].split(',')[3];
    const paintsColor = alpha === undefined ? cs.backgroundColor !== 'transparent' : parseFloat(alpha) > 0;
    if (paintsColor || cs.backgroundImage !== 'none') {
      console.error('[NativeSlot] this element paints a background over the native preview:', el, cs.backgroundColor, cs.backgroundImage);
      el.style.outline = '2px solid red';
    }
  }
}
