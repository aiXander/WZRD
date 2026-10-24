// Studio's preview monitor (S10, Resolume-style): the selected item plays
// in one place.
// - HAP → natively: the engine decodes it into the `studio` underlay slot
//   (`preview.file`, P1) — no leg renders for it, LIVE is untouched.
// - mp4 / webm → the webview's <video> (WebKit decodes H.264 in hardware).
// - stills → <img>.
// - DXV and anything else → its thumbnail + "Make playable to preview".
// Closing (unmount, or nothing selected) releases the engine's decode
// thread: `preview.file null`, and the slot leaving file mode drops it too.

import { useEffect, useState } from 'react';
import { mediaUrl, type LibraryItem } from '../../api/hub';
import { previewFile } from '../../api/ipc';
import { nativeUnderlay } from '../../api/previewMode';
import { itemByPath, useStudio } from '../../state/studio';
import { NativeSlot } from '../NativeSlot';

type Mode = 'native' | 'video' | 'image' | 'thumb' | 'empty';

function modeOf(it: LibraryItem | undefined): Mode {
  if (!it) return 'empty';
  if (it.kind === 'image') return 'image';
  if (it.codec === 'hap') return nativeUnderlay() ? 'native' : 'thumb';
  if (it.codec === 'h264' || it.codec === 'hevc' || /\.(mp4|m4v|webm)$/i.test(it.path)) return 'video';
  return 'thumb';
}

export function Monitor() {
  const path = useStudio((s) => s.monitor);
  // Re-read on listing refresh (labels change the object, not the file).
  const item = useStudio((s) => (s.monitor ? s.listing?.items.find((i) => i.path === s.monitor) : undefined));
  const mode = modeOf(item ?? itemByPath(path));
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (mode !== 'native' || !path) return;
    setErr(null);
    let live = true;
    previewFile(path).catch((e) => live && setErr(String(e)));
    return () => {
      live = false;
      previewFile(null).catch(() => {});
    };
  }, [mode, path]);

  return (
    <section className="relative shrink-0 border border-line rounded-lg overflow-hidden" style={{ aspectRatio: '16 / 9' }}>
      {mode === 'native' && (
        <NativeSlot id="studio" className="absolute inset-0">
          {err && <div className="absolute bottom-1 left-1 right-1 text-[10px] text-bad bg-black/70 px-1">{err}</div>}
        </NativeSlot>
      )}
      {mode !== 'native' && <div className="absolute inset-0 bg-black" />}
      {mode === 'video' && item && (
        <video key={item.path} src={mediaUrl(item.path)} autoPlay loop muted playsInline className="absolute inset-0 w-full h-full object-contain" />
      )}
      {mode === 'image' && item && (
        <img src={mediaUrl(item.path)} className="absolute inset-0 w-full h-full object-contain" />
      )}
      {mode === 'thumb' && item && (
        <>
          {item.thumb && <img src={mediaUrl(item.thumb)} className="absolute inset-0 w-full h-full object-contain opacity-60" />}
          <div className="absolute inset-x-0 bottom-0 p-2 text-[11px] text-fg2 bg-black/60">
            {item.codec === 'hap'
              ? 'HAP plays natively in the app window.'
              : `${(item.codec ?? 'This file').toUpperCase()} — Make playable to preview.`}
          </div>
        </>
      )}
      {mode === 'empty' && (
        <div className="absolute inset-0 flex items-center justify-center text-[11px] text-fg3">
          Click an item to show it here
        </div>
      )}
      <span className="absolute top-1 left-1 text-[9px] font-bold tracking-wider px-1 rounded bg-black/60 text-fg3">
        MONITOR
      </span>
    </section>
  );
}
