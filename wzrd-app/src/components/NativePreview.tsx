// The monitor. Measures the largest pack-aspect rect that fits its slot and
// shows the native preview there (lossless, full-rate blit of the engine
// composite, `<NativeSlot>` with Perform's default params: the deck-toggle
// leg). With the underlay, HTML can draw over it; with `WZRD_PREVIEW=window`
// the child window covers the rect, so nothing can.
//
// In a plain browser (dev bridge) the slot shows the JPEG `preview` channel
// instead — the design composite.

import { useState } from 'react';
import { previewMode } from '../api/previewMode';
import { useStore } from '../state/store';
import { useAspectFit } from '../ui/useAspectFit';
import { usePreviewJpegDemand } from '../ui/usePreviewJpegDemand';
import { NativeSlot } from './NativeSlot';

export function NativePreview({ className = '' }: { className?: string }) {
  const [outer, setOuter] = useState<HTMLDivElement | null>(null);
  const pack = useStore((s) => s.pack);
  // The largest aspect-correct rect inside the slot, either dimension.
  const box = useAspectFit(outer, pack ? pack.width / pack.height : 16 / 9);
  const native = previewMode() !== 'none';

  return (
    <div ref={setOuter} className={'relative w-full h-full min-h-0 flex items-center justify-center ' + className}>
      <NativeSlot
        native={native}
        className="relative flex items-center justify-center"
        style={{ width: box.w, height: box.h }}
      >
        {!native && <JpegFeed />}
      </NativeSlot>
    </div>
  );
}

/** The ~15 fps JPEG `preview` channel (always the design composite). */
export function JpegFeed({ className = '' }: { className?: string }) {
  const preview = useStore((s) => s.preview);
  usePreviewJpegDemand(true);
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
