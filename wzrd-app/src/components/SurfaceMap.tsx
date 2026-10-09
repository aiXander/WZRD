// The surface map — the photographed surface with its named regions, the
// primary visual of the Prepare room (spec §1: "the surface map is primary").
//
// Backdrop: PHOTO (the pack's surface.png), LIGHT (the design composite, the
// ~15 fps JPEG feed) or BOTH (light over a dimmed photo — see exactly where
// light lands on the real thing). Regions paint in their own colour; the
// selected layer's regions glow, the hovered region gets a floating label.

import { useEffect, useMemo, useRef, useState } from 'react';
import { useAspectFit } from '../ui/useAspectFit';
import type { PackInfo } from '../api/ipc';
import { getMask, getSurface, pickRegion, useMaskVersion } from '../state/masks';
import { regionName, regionsOf, selectorCovers, type Binding } from '../state/scene';
import { useStore } from '../state/store';
import { Seg } from '../ui/primitives';

type Backdrop = 'photo' | 'light' | 'both';

export function SurfaceMap({
  bindings,
  onRegionClick,
  armed,
}: {
  bindings: Binding[];
  onRegionClick: (regionId: string | null) => void;
  /** "Click a region to create a layer" mode — crosshair + hint. */
  armed?: boolean;
}) {
  const pack = useStore((s) => s.pack);
  const preview = useStore((s) => s.preview);
  const selectedBinding = useStore((s) => s.selectedBindingId);
  const hoveredBinding = useStore((s) => s.hoveredBindingId);
  const selectedRegion = useStore((s) => s.selectedLayerId);
  const maskV = useMaskVersion();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [wrap, setWrap] = useState<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const [backdrop, setBackdrop] = useState<Backdrop>('both');
  const [overlays, setOverlays] = useState(true);
  const [img, setImg] = useState<HTMLImageElement | null>(null);


  // Decode JPEG frames off the paint path.
  useEffect(() => {
    if (!preview || backdrop === 'photo') return;
    let cancelled = false;
    const im = new Image();
    im.src = `data:image/jpeg;base64,${preview.data_b64}`;
    im.decode()
      .then(() => !cancelled && setImg(im))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [preview, backdrop]);

  // Fit the canvas to the slot at pack aspect.
  const box = useAspectFit(wrap, pack ? pack.width / pack.height : 4 / 3);

  const sel = useMemo(() => byBinding(bindings, selectedBinding, pack), [bindings, selectedBinding, pack]);
  const hot = useMemo(() => {
    const s = byBinding(bindings, hoveredBinding, pack);
    if (hover) s.add(hover.id);
    return s;
  }, [bindings, hoveredBinding, hover, pack]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !pack) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    if (c.width !== pack.width || c.height !== pack.height) {
      c.width = pack.width;
      c.height = pack.height;
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, pack.width, pack.height);
    const photo = getSurface();
    if (photo && backdrop !== 'light') {
      ctx.globalAlpha = backdrop === 'both' ? 0.42 : 1;
      ctx.drawImage(photo, 0, 0, pack.width, pack.height);
    }
    if (img && backdrop !== 'photo') {
      // Light adds onto the surface — exactly what the projector does.
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = backdrop === 'both' ? 'lighter' : 'source-over';
      ctx.drawImage(img, 0, 0, pack.width, pack.height);
      ctx.globalCompositeOperation = 'source-over';
    }
    if (overlays) {
      const any = sel.size > 0 || hot.size > 0 || !!selectedRegion;
      const tier = (id: string) => (hot.has(id) ? 2 : sel.has(id) || id === selectedRegion ? 1 : 0);
      const alpha = [any ? 0.08 : 0.2, 0.5, 0.7];
      for (const t of [0, 1, 2]) {
        for (const l of pack.layers) {
          if (tier(l.id) !== t) continue;
          const m = getMask(l.id);
          if (!m) continue;
          ctx.globalAlpha = alpha[t];
          ctx.drawImage(m.overlay, 0, 0);
        }
      }
      ctx.globalAlpha = 1;
    }
  }, [pack, img, backdrop, overlays, sel, hot, selectedRegion, maskV]);

  if (!pack) return <div className="p-4 text-[12px] text-fg3">Waiting for the layer pack…</div>;

  const uvOf = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height] as const;
  };
  const coveredBy = (id: string) =>
    bindings.filter((b) => {
      const l = pack.layers.find((x) => x.id === id);
      return l && selectorCovers(b.select, pack, l);
    });

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-2 h-9 px-2 border-b border-line shrink-0">
        <Seg<Backdrop>
          value={backdrop}
          onChange={setBackdrop}
          options={[
            { id: 'photo', label: 'Photo', title: 'The photographed surface' },
            { id: 'light', label: 'Light', title: 'The design composite (what TAKE would project)' },
            { id: 'both', label: 'Both', title: 'Light added onto the photo — where it lands on the real thing' },
          ]}
        />
        <button
          onClick={() => setOverlays((o) => !o)}
          className={
            'h-[26px] px-2 rounded-md border text-[11px] ' +
            (overlays ? 'border-line2 bg-raised text-fg' : 'border-line text-fg3 hover:text-fg')
          }
          title="Tint each region in its colour"
        >
          Regions
        </button>
        <div className="flex-1" />
        <span className="text-[10px] text-fg3">
          {pack.layers.length} regions · {pack.width}×{pack.height}
        </span>
      </div>
      <div ref={setWrap} className="relative flex-1 min-h-0 flex items-center justify-center p-3 dim-grid">
        <div className="relative" style={{ width: box.w, height: box.h }}>
          <canvas
            ref={canvasRef}
            className={'w-full h-full rounded-md ring-1 ring-line ' + (armed ? 'cursor-crosshair' : 'cursor-pointer')}
            onMouseMove={(e) => {
              const [u, v] = uvOf(e);
              const id = pickRegion(pack, u, v);
              const r = canvasRef.current!.getBoundingClientRect();
              setHover(id ? { id, x: e.clientX - r.left, y: e.clientY - r.top } : null);
            }}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const [u, v] = uvOf(e);
              onRegionClick(pickRegion(pack, u, v));
            }}
          />
          {hover && (
            <div
              className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[calc(100%+12px)] whitespace-nowrap rounded-md border border-line2 bg-raised/95 px-2 py-1 shadow-xl"
              style={{ left: hover.x, top: hover.y }}
            >
              <div className="text-[12px] font-medium">{regionName(pack, hover.id)}</div>
              <div className="text-[10px] text-fg3">
                {(() => {
                  const c = coveredBy(hover.id);
                  return armed
                    ? 'click to create a layer here'
                    : c.length
                    ? `lit by ${c.map((b) => b.id).join(', ')}`
                    : 'dark — click to give it a layer';
                })()}
              </div>
            </div>
          )}
          {armed && (
            <div className="pointer-events-none absolute top-2 left-1/2 -translate-x-1/2 rounded-full bg-design text-black text-[11px] font-semibold px-3 py-1 shadow-lg">
              Click a region to give it a new layer
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function byBinding(bindings: Binding[], id: string | null, pack: PackInfo | null): Set<string> {
  const b = bindings.find((x) => x.id === id);
  return new Set(regionsOf(b, pack).map((l) => l.id));
}
