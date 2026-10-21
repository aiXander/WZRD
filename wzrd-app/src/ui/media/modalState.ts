// "A modal is open" (show-setup G11) + who gets Esc.
//
// - `modalOpen()` is a refcount: every mounted Modal / Lightbox holds it
//   (`useModalLock`). `state/shortcuts.ts` and Studio's key handler return
//   early while it's > 0, so digits in a lightbox never score or select
//   layers and T never TAKEs.
// - Esc goes to the **topmost** layer only (`useEscapeLayer`): one capture
//   listener on window calls the newest layer's handler and stops the event,
//   so a Lightbox over a Modal closes alone. An open `<Popover>` (it marks
//   itself `data-popover`) wins over every layer — its own window listener
//   closes it, the next Esc reaches the layer.

import { useEffect, useRef } from 'react';

let locks = 0;

export function modalOpen(): boolean {
  return locks > 0;
}

/** Hold the "a modal is open" flag while mounted (and `active`). */
export function useModalLock(active = true): void {
  useEffect(() => {
    if (!active) return;
    locks += 1;
    const body = document.body.style;
    const prev = body.overflow;
    body.overflow = 'hidden';
    return () => {
      locks -= 1;
      body.overflow = prev;
    };
  }, [active]);
}

type Layer = { id: number; fn: () => void };
const layers: Layer[] = [];
let layerSeq = 0;
let installed = false;

function onKeyCapture(e: KeyboardEvent) {
  if (e.key !== 'Escape' || !layers.length) return;
  if (document.querySelector('[data-popover]')) return; // the popover closes first
  e.stopPropagation();
  e.preventDefault();
  layers[layers.length - 1].fn();
}

/** Register an Esc handler on top of the layer stack while mounted. */
export function useEscapeLayer(onEscape: () => void, active = true): void {
  const fn = useRef(onEscape);
  fn.current = onEscape;
  useEffect(() => {
    if (!active) return;
    if (!installed) {
      window.addEventListener('keydown', onKeyCapture, true);
      installed = true;
    }
    const layer: Layer = { id: ++layerSeq, fn: () => fn.current() };
    layers.push(layer);
    return () => {
      const i = layers.findIndex((l) => l.id === layer.id);
      if (i >= 0) layers.splice(i, 1);
    };
  }, [active]);
}
