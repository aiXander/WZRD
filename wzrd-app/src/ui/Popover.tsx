// Floating panel anchored to an element, rendered in a portal so scrolling
// panels can't clip it. Flips above the anchor when there's no room below.
// Closes on outside press or Escape.

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export function Popover({
  anchor,
  onClose,
  width = 260,
  children,
}: {
  anchor: HTMLElement | null;
  onClose: () => void;
  width?: number;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!anchor || !el) return;
    const place = () => {
      const a = anchor.getBoundingClientRect();
      const h = el.offsetHeight;
      const below = a.bottom + 6;
      const top = below + h > window.innerHeight - 8 ? Math.max(8, a.top - h - 6) : below;
      const left = Math.min(window.innerWidth - width - 8, Math.max(8, a.right - width));
      setPos((p) => (p && p.left === left && p.top === top ? p : { left, top }));
    };
    place();
    // Content that grows while open (e.g. the ModEditor revealing shape +
    // rate after a source pick) must not run off the bottom of the window.
    const ro = new ResizeObserver(place);
    ro.observe(el);
    return () => ro.disconnect();
  }, [anchor, width]);

  useEffect(() => {
    const h = (e: PointerEvent) => {
      const t = e.target as Node;
      if (ref.current?.contains(t) || anchor?.contains(t)) return;
      onClose();
    };
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('pointerdown', h, true);
    window.addEventListener('keydown', k);
    return () => {
      window.removeEventListener('pointerdown', h, true);
      window.removeEventListener('keydown', k);
    };
  }, [onClose, anchor]);

  return createPortal(
    <div
      ref={ref}
      // ui/media/modalState.ts: an open popover gets Esc before the modal under it.
      data-popover=""
      style={{ width, left: pos?.left ?? -9999, top: pos?.top ?? -9999 }}
      className="fixed z-[60] rounded-lg border border-line2 bg-raised shadow-[0_16px_48px_rgba(0,0,0,.6)] p-2.5"
    >
      {children}
    </div>,
    document.body
  );
}
