// A centred dialog in a portal (show-setup M1): dim backdrop, title bar
// (+ `headerRight`), scrolling body, optional footer. Esc closes unless a
// Lightbox or Popover on top of it takes the key (modalState's layer stack);
// the backdrop closes only with `closeOnBackdrop`. While open it holds
// `modalOpen()`, so room shortcuts stay quiet (G11).
//
// Portalled to <body>, so it's a sibling of the page and never paints over
// a <NativeSlot> ancestor (the hole rule) — it simply covers the slot while
// open.

import { createPortal } from 'react-dom';
import { useEscapeLayer, useModalLock } from './modalState';

export function Modal({
  title,
  onClose,
  width = 1080,
  height = 'min(86vh, 900px)',
  headerRight,
  footer,
  closeOnBackdrop = false,
  children,
}: {
  title: React.ReactNode;
  onClose: () => void;
  width?: number | string;
  height?: string;
  headerRight?: React.ReactNode;
  footer?: React.ReactNode;
  closeOnBackdrop?: boolean;
  children: React.ReactNode;
}) {
  useModalLock();
  useEscapeLayer(onClose);
  return createPortal(
    <div
      className="fixed inset-0 z-[52] bg-black/60 flex items-center justify-center p-4"
      onMouseDown={(e) => {
        if (closeOnBackdrop && e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        className="flex flex-col min-h-0 max-w-full max-h-full rounded-xl border border-line2 bg-panel shadow-[0_24px_80px_rgba(0,0,0,.7)] animate-in"
        style={{ width: typeof width === 'number' ? `${width}px` : width, height }}
      >
        <header className="flex items-center gap-2 h-10 px-4 border-b border-line shrink-0 min-w-0">
          <div className="text-[13px] font-semibold text-fg truncate min-w-0">{title}</div>
          <div className="flex-1" />
          {headerRight}
          <button
            className="ml-1 w-7 h-7 rounded-md text-fg3 hover:text-fg hover:bg-raised text-[14px]"
            onClick={onClose}
            title="Close (Esc)"
          >
            ✕
          </button>
        </header>
        <div className="flex-1 min-h-0 overflow-y-auto">{children}</div>
        {footer && (
          <footer className="flex items-center gap-2 px-4 py-2.5 border-t border-line shrink-0">{footer}</footer>
        )}
      </div>
    </div>,
    document.body
  );
}
