// Right-click menu at the pointer, rendered in a portal so scrolling panels
// can't clip it. Kept inside the window; closes on outside press, Escape,
// scroll or window blur.

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export type MenuItem =
  | { label: string; onClick: () => void; danger?: boolean; disabled?: boolean; checked?: boolean; title?: string }
  | { heading: string }
  | 'separator';

export function ContextMenu({
  x,
  y,
  items,
  onClose,
}: {
  x: number;
  y: number;
  items: MenuItem[];
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const left = Math.max(8, Math.min(x, window.innerWidth - el.offsetWidth - 8));
    const top = Math.max(8, Math.min(y, window.innerHeight - el.offsetHeight - 8));
    setPos({ left, top });
  }, [x, y]);

  useEffect(() => {
    const press = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose();
    };
    const key = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    const scroll = (e: Event) => {
      if (!ref.current?.contains(e.target as Node)) onClose();
    };
    window.addEventListener('pointerdown', press, true);
    window.addEventListener('keydown', key);
    window.addEventListener('scroll', scroll, true);
    window.addEventListener('blur', onClose);
    return () => {
      window.removeEventListener('pointerdown', press, true);
      window.removeEventListener('keydown', key);
      window.removeEventListener('scroll', scroll, true);
      window.removeEventListener('blur', onClose);
    };
  }, [onClose]);

  return createPortal(
    <div
      ref={ref}
      // ui/media/modalState.ts: an open popover gets Esc before the modal under it.
      data-popover=""
      role="menu"
      style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999 }}
      className="fixed z-[70] font-sans min-w-[11rem] max-w-[16rem] rounded-lg border border-line2 bg-raised py-1 text-[12px] shadow-[0_16px_48px_rgba(0,0,0,.6)]"
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((it, i) =>
        it === 'separator' ? (
          <div key={i} className="my-1 border-t border-line" />
        ) : 'heading' in it ? (
          <div key={i} className="px-3 pt-1.5 pb-0.5 text-[10px] uppercase tracking-wide text-fg3">
            {it.heading}
          </div>
        ) : (
          <button
            key={i}
            role="menuitem"
            type="button"
            title={it.title}
            disabled={it.disabled}
            onClick={() => {
              onClose();
              it.onClick();
            }}
            className={
              'flex w-full items-center gap-2 px-3 py-1.5 text-left disabled:opacity-40 ' +
              (it.danger ? 'text-bad hover:bg-bad/15' : 'text-fg hover:bg-panel2')
            }
          >
            <span className="w-3 shrink-0 text-center text-[11px]">{it.checked ? '✓' : it.danger ? '✕' : ''}</span>
            <span className="truncate">{it.label}</span>
          </button>
        )
      )}
    </div>,
    document.body
  );
}
