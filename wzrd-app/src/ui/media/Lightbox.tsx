// Full-window viewer over a list of media (show-setup M1): an image or a
// looping `<video controls>`. Keys: ← → step · Esc closes (only the
// lightbox — a Modal under it stays; modalState's layer stack) · F toggles
// fit / 1:1 (drag pans at 1:1) · Space = play/pause a video, or, with
// `compareWith`, hold to flip the blend. Top bar: name · W×H · size, the
// caller's `actions(ref)` (pin, delete …), fit toggle, close.
//
// `compareWith` draws another image in the same box under the item, with a
// blend slider for how much of the item shows (e.g. the dark surface over
// the lit photo it came from).

import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { mediaUrl } from '../../api/hub';
import { useEscapeLayer, useModalLock } from './modalState';
import { baseName, fmtBytes, kindOf, type MediaRef } from './types';

function typing(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  return t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable;
}

export function Lightbox({
  items,
  index,
  onIndex,
  onClose,
  compareWith,
  actions,
}: {
  items: MediaRef[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
  compareWith?: string | null;
  actions?: (ref: MediaRef) => React.ReactNode;
}) {
  useModalLock();
  useEscapeLayer(onClose);
  const i = Math.max(0, Math.min(items.length - 1, index));
  const m = items[i] as MediaRef | undefined;
  const kind = m ? kindOf(m) : null;
  // HAP / DXV don't decode in a <video>: show the still (Studio's monitor
  // plays HAP natively).
  const codec = m?.item?.codec ?? null;
  const webVideo = kind === 'video' && !(codec && /^(hap|dxv)/i.test(codec));
  const [oneToOne, setOneToOne] = useState(false);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [blend, setBlend] = useState(0.5);
  const [flip, setFlip] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);

  // The list emptied under us (last item deleted) → nothing to show.
  useEffect(() => {
    if (!items.length) onClose();
  }, [items.length]);

  useEffect(() => {
    setPan({ x: 0, y: 0 });
    setNatural(null);
  }, [m?.path]);

  // Latest values for the window listeners (registered once).
  const st = useRef({ i, n: items.length, onIndex, kind, compare: !!compareWith });
  st.current = { i, n: items.length, onIndex, kind, compare: !!compareWith };

  useEffect(() => {
    // The key that opened the viewer (Studio's F / Enter) can still reach
    // this listener — ignore keys in the first moment after mount.
    const openedAt = performance.now();
    const down = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.repeat || performance.now() - openedAt < 200) return;
      const s = st.current;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        if (s.n > 1) s.onIndex((s.i + (e.key === 'ArrowRight' ? 1 : -1) + s.n) % s.n);
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        setOneToOne((v) => !v);
        setPan({ x: 0, y: 0 });
      } else if (e.key === ' ') {
        if (e.target instanceof HTMLVideoElement) return; // its own controls
        e.preventDefault();
        // A just-clicked button (Fit, an action) would also fire on Space's keyup.
        if (document.activeElement instanceof HTMLButtonElement) document.activeElement.blur();
        if (s.compare) setFlip(true);
        else if (s.kind === 'video' && videoRef.current) {
          const v = videoRef.current;
          if (v.paused) v.play().catch(() => {});
          else v.pause();
        }
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === ' ') setFlip(false);
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, []);

  if (!m) return null;
  const w = m.w ?? natural?.w ?? null;
  const h = m.h ?? natural?.h ?? null;
  const size = m.item?.size ?? null;
  const shown = flip ? 1 - blend : blend;
  const mainOpacity = compareWith ? shown : 1;
  // Fit: item and comparison both fill the box with `contain`, so same-aspect
  // images line up exactly. 1:1: natural size, the comparison scaled into it.
  const fitCls = oneToOne ? 'max-w-none max-h-none' : 'w-full h-full object-contain';

  const onPointerDown = (e: React.PointerEvent) => {
    if (!oneToOne || e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    setPan({ x: d.px + e.clientX - d.x, y: d.py + e.clientY - d.y });
  };
  const onPointerUp = () => (drag.current = null);

  const step = (d: number) => items.length > 1 && onIndex((i + d + items.length) % items.length);

  return createPortal(
    <div className="fixed inset-0 z-[58] bg-black/95 flex flex-col select-none">
      <header className="flex items-center gap-3 h-10 px-3 border-b border-line bg-panel/90 shrink-0 min-w-0">
        <span className="text-[12px] text-fg truncate min-w-0" title={m.path}>
          {m.name ?? baseName(m.path)}
        </span>
        <span className="text-[11px] text-fg3 num shrink-0">
          {[w && h ? `${w}×${h}` : null, fmtBytes(size) || null, items.length > 1 ? `${i + 1} / ${items.length}` : null]
            .filter(Boolean)
            .join(' · ')}
        </span>
        <div className="flex-1" />
        {compareWith && (
          <label
            className="flex items-center gap-1.5 text-[10px] text-fg3 shrink-0"
            title="How much of this item shows over the comparison image (hold Space to flip)"
          >
            compare
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={blend}
              onChange={(e) => setBlend(Number(e.target.value))}
              onKeyDown={(e) => e.key !== 'Escape' && e.stopPropagation()}
              className="w-28 accent-[rgb(var(--deck))]"
            />
            <span className="num w-7 text-fg2">{Math.round(shown * 100)}%</span>
          </label>
        )}
        {actions && <div className="flex items-center gap-1 shrink-0">{actions(m)}</div>}
        <button
          className={
            'h-6 px-2 rounded-md border text-[11px] shrink-0 ' +
            (oneToOne ? 'border-deck/70 text-fg bg-deck/20' : 'border-line text-fg2 hover:text-fg')
          }
          onClick={() => {
            setOneToOne((v) => !v);
            setPan({ x: 0, y: 0 });
          }}
          title="Fit / 1:1 (F) — drag to pan at 1:1"
        >
          {oneToOne ? '1:1' : 'Fit'}
        </button>
        <button
          className="w-7 h-7 rounded-md text-fg3 hover:text-fg hover:bg-raised text-[14px] shrink-0"
          onClick={onClose}
          title="Close (Esc)"
        >
          ✕
        </button>
      </header>
      <div
        className={'relative flex-1 min-h-0 overflow-hidden flex items-center justify-center p-4 ' + (oneToOne ? 'cursor-grab active:cursor-grabbing' : '')}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => kind === 'image' && setOneToOne((v) => !v)}
      >
        <div
          className="relative"
          style={oneToOne ? { transform: `translate(${pan.x}px, ${pan.y}px)` } : { width: '100%', height: '100%' }}
        >
          {compareWith && (
            <img
              src={mediaUrl(compareWith)}
              draggable={false}
              className="absolute inset-0 w-full h-full object-contain"
            />
          )}
          {kind === 'video' && !webVideo ? (
            <div className="flex flex-col items-center gap-2 w-full h-full justify-center">
              {m.thumb && <img src={mediaUrl(m.thumb)} draggable={false} className="max-w-full max-h-[85%] object-contain" />}
              <span className="text-[11px] text-fg3">
                {(codec ?? 'video').toUpperCase()} — plays in the monitor, not in this viewer
              </span>
            </div>
          ) : kind === 'video' ? (
            <video
              key={m.path}
              ref={videoRef}
              src={mediaUrl(m.path)}
              controls
              loop
              autoPlay
              playsInline
              onLoadedMetadata={(e) =>
                setNatural({ w: e.currentTarget.videoWidth, h: e.currentTarget.videoHeight })
              }
              className={'relative ' + fitCls}
              style={{ opacity: mainOpacity }}
            />
          ) : kind === 'image' ? (
            <img
              key={m.path}
              src={mediaUrl(m.path)}
              draggable={false}
              onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
              className={'relative ' + fitCls}
              style={{ opacity: mainOpacity }}
            />
          ) : (
            <div className="text-[12px] text-fg3">
              {m.item?.codec ? `${m.item.codec.toUpperCase()} can't be shown here` : 'No preview for this file'}
            </div>
          )}
        </div>
        {items.length > 1 && (
          <>
            <NavBtn side="left" onClick={() => step(-1)} />
            <NavBtn side="right" onClick={() => step(1)} />
          </>
        )}
      </div>
    </div>,
    document.body
  );
}

function NavBtn({ side, onClick }: { side: 'left' | 'right'; onClick: () => void }) {
  return (
    <button
      onPointerDown={(e) => e.stopPropagation()}
      onClick={onClick}
      title={side === 'left' ? 'Previous (←)' : 'Next (→)'}
      className={
        'absolute top-1/2 -translate-y-1/2 w-10 h-16 rounded-lg bg-black/40 text-fg2 hover:text-fg hover:bg-black/70 text-[22px] ' +
        (side === 'left' ? 'left-3' : 'right-3')
      }
    >
      {side === 'left' ? '‹' : '›'}
    </button>
  );
}
