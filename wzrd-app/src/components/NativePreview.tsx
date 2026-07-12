// Native preview slot (app-collapse Step 3). Renders an empty, measured
// placeholder div; the Rust side positions a borderless native child window
// over it that blits the engine's composite texture directly — lossless,
// full-resolution, full-rate, unlike the ~15 fps JPEG thumbnail path
// (which survives for the Prepare canvas underlay + remote WS clients).
//
// Position sync: rect pushed via `preview_set_bounds` on mount, resize
// (ResizeObserver + window resize) and scroll (capture phase). Window
// drags need no re-push — the preview is a macOS child window of the main
// window and moves with it.

import { useEffect, useRef } from 'react';
import { previewSetBounds } from '../api/ipc';
import { useStore } from '../state/store';

export function NativePreview() {
  const ref = useRef<HTMLDivElement | null>(null);
  const pack = useStore((s) => s.pack);
  const aspect = pack ? `${pack.width} / ${pack.height}` : '16 / 9';

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const push = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        previewSetBounds(
          r.left,
          r.top,
          r.width,
          r.height,
          r.width > 4 && r.height > 4
        ).catch(() => {});
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
  }, []);

  return (
    <div className="w-full h-full flex items-center justify-center min-h-0">
      <div
        ref={ref}
        className="max-w-full max-h-full h-full rounded border border-ink-600 bg-ink-900 flex items-center justify-center text-xs text-zinc-500"
        style={{ aspectRatio: aspect }}
      >
        native preview attaching…
      </div>
    </div>
  );
}
