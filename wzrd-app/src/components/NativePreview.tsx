// The monitor. Measures the largest pack-aspect rect that fits its slot and
// positions the native preview child window exactly over it (lossless,
// full-rate blit of the engine composite — `preview_set_bounds`). HTML can't
// draw over that rect (the native window covers it), so all chrome lives
// around it, never on it.
//
// In a plain browser (dev bridge, no native window) the slot shows the JPEG
// `preview` channel instead — the design composite.

import { useEffect, useRef, useState } from 'react';
import { previewSetBounds } from '../api/ipc';
import { IN_TAURI } from '../api/transport';
import { useStore } from '../state/store';
import { useAspectFit } from '../ui/useAspectFit';

export function NativePreview({ className = '' }: { className?: string }) {
  const [outer, setOuter] = useState<HTMLDivElement | null>(null);
  const inner = useRef<HTMLDivElement | null>(null);
  const pack = useStore((s) => s.pack);
  // The largest aspect-correct rect inside the slot, either dimension.
  const box = useAspectFit(outer, pack ? pack.width / pack.height : 16 / 9);

  useEffect(() => {
    if (!IN_TAURI) return;
    const el = inner.current;
    if (!el) return;
    let raf = 0;
    const push = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        previewSetBounds(r.left, r.top, r.width, r.height, r.width > 4 && r.height > 4).catch(() => {});
      });
    };
    push();
    const ro = new ResizeObserver(push);
    ro.observe(el);
    window.addEventListener('resize', push);
    window.addEventListener('scroll', push, true);
    // Re-push on focus: heals child-window detachment after minimize /
    // display changes (the backend re-attaches on every bounds push).
    window.addEventListener('focus', push);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener('resize', push);
      window.removeEventListener('scroll', push, true);
      window.removeEventListener('focus', push);
      previewSetBounds(0, 0, 0, 0, false).catch(() => {});
    };
  }, [box.w, box.h]);

  return (
    <div ref={setOuter} className={'relative w-full h-full min-h-0 flex items-center justify-center ' + className}>
      <div
        ref={inner}
        className="relative bg-black flex items-center justify-center"
        style={{ width: box.w, height: box.h }}
      >
        {IN_TAURI ? (
          <span className="text-2xs text-fg4">attaching preview…</span>
        ) : (
          <JpegFeed />
        )}
      </div>
    </div>
  );
}

/** The ~15 fps JPEG `preview` channel (always the design composite). */
export function JpegFeed({ className = '' }: { className?: string }) {
  const preview = useStore((s) => s.preview);
  if (!preview) return <span className="text-2xs text-fg4">no preview frames</span>;
  return (
    <img
      alt=""
      draggable={false}
      className={'w-full h-full object-contain ' + className}
      src={`data:image/jpeg;base64,${preview.data_b64}`}
    />
  );
}
