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

export function NativeSlot({
  params,
  native = previewMode() === 'underlay',
  className = '',
  style,
  children,
}: {
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
    const push = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        previewSetSlot({
          x: r.left,
          y: r.top,
          width: r.width,
          height: r.height,
          visible: r.width > 4 && r.height > 4,
          params: p,
        }).catch(() => {});
      });
    };
    push();
    const ro = new ResizeObserver(push);
    ro.observe(el);
    window.addEventListener('resize', push);
    window.addEventListener('scroll', push, true);
    // The child window (`window` mode) detaches after minimize / display
    // changes; the backend re-attaches on every push.
    if (previewMode() === 'window') window.addEventListener('focus', push);
    if (import.meta.env.DEV && previewMode() === 'underlay') checkHole(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', push);
      window.removeEventListener('scroll', push, true);
      window.removeEventListener('focus', push);
      previewSetSlot({ x: 0, y: 0, width: 0, height: 0, visible: false }).catch(() => {});
    };
  }, [native, paramsKey]);

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
