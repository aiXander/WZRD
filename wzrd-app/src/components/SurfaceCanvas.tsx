// Surface canvas — the photo + mask overlays + named regions.
//
// 4.2 design: the surface canvas is the primary visual on every page that has
// one (per `user_design_spec.md`). Phase 4.2 lands the read-only version —
// pan + zoom only, no renaming, no sidecar identity.json editing. Renaming
// stays deferred to 4.3+.
//
// Implementation: an HTML <canvas> sized to the pack resolution, masks
// loaded as base64 PNGs via the Tauri `read_mask_png` command and drawn
// with reduced alpha tinted per layer. The preview JPEG sits underneath as
// the "real" backing so the surface canvas is *also* a live performance
// view when the route is mounted.

import { useEffect, useMemo, useRef, useState } from 'react';
import { readMaskPng } from '../api/ipc';
import { useStore } from '../state/store';

// One stable hue per layer index; same colour comes back across renders.
function hueForIndex(i: number): number {
  return (i * 47) % 360;
}

export function SurfaceCanvas() {
  const pack = useStore((s) => s.pack);
  const preview = useStore((s) => s.preview);
  const selected = useStore((s) => s.selectedBindingId);
  const setSelected = useStore((s) => s.setSelectedBindingId);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [masks, setMasks] = useState<Record<string, HTMLImageElement>>({});
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [showOverlays, setShowOverlays] = useState(true);

  // Lazy-load each layer's mask PNG once.
  useEffect(() => {
    if (!pack) return;
    let cancelled = false;
    (async () => {
      for (const layer of pack.layers) {
        if (masks[layer.id]) continue;
        try {
          const b64 = await readMaskPng(layer.mask_path);
          if (cancelled) return;
          const img = new Image();
          img.src = `data:image/png;base64,${b64}`;
          await img.decode().catch(() => {});
          if (cancelled) return;
          setMasks((m) => ({ ...m, [layer.id]: img }));
        } catch (e) {
          console.warn('mask load', layer.id, e);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pack]);

  // Repaint on every preview/pack/selection change.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !pack) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

    c.width = pack.width;
    c.height = pack.height;
    ctx.fillStyle = '#08090b';
    ctx.fillRect(0, 0, pack.width, pack.height);

    // Live preview underneath.
    if (preview) {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${preview.data_b64}`;
      // We can't await here — accept that the first paint may miss this
      // frame and the next telemetry tick will fix it.
      img.decode().then(() => {
        ctx.drawImage(img, 0, 0, pack.width, pack.height);
        drawOverlays();
      }).catch(() => drawOverlays());
    } else {
      drawOverlays();
    }

    function drawOverlays() {
      if (!showOverlays || !pack) return;
      pack.layers.forEach((layer, i) => {
        const m = masks[layer.id];
        if (!m) return;
        const hue = hueForIndex(i);
        const tint = `hsla(${hue}, 70%, 60%, 0.25)`;
        ctx!.save();
        ctx!.globalCompositeOperation = 'source-over';
        // Mask-as-alpha trick: draw mask, set composite to source-in to
        // colorize, then back to source-over.
        ctx!.globalAlpha = 1.0;
        ctx!.drawImage(m, 0, 0);
        ctx!.globalCompositeOperation = 'source-in';
        ctx!.fillStyle = tint;
        ctx!.fillRect(0, 0, pack!.width, pack!.height);
        ctx!.restore();

        if (selected === layer.id || hoverId === layer.id) {
          // Brighter outline. Cheap pass: draw the mask edge again at higher
          // alpha. Without a per-pixel edge detect this just thickens the
          // overlay; good enough for "which region is highlighted."
          ctx!.save();
          ctx!.globalCompositeOperation = 'source-over';
          ctx!.globalAlpha = 0.45;
          ctx!.drawImage(m, 0, 0);
          ctx!.globalCompositeOperation = 'source-in';
          ctx!.fillStyle = `hsla(${hue}, 90%, 70%, 0.65)`;
          ctx!.fillRect(0, 0, pack!.width, pack!.height);
          ctx!.restore();
        }

        if (layer.centroid) {
          ctx!.fillStyle = '#fff';
          ctx!.font = '12px ui-monospace';
          ctx!.fillText(layer.id, layer.centroid[0] + 4, layer.centroid[1]);
        }
      });
    }
  }, [pack, preview, masks, selected, hoverId, showOverlays]);

  const layers = useMemo(() => pack?.layers ?? [], [pack]);

  function pickAtClient(evt: React.MouseEvent<HTMLCanvasElement>): string | null {
    if (!pack) return null;
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * pack.width;
    const y = ((evt.clientY - rect.top) / rect.height) * pack.height;
    // Walk in reverse so topmost (visually-last-drawn) wins.
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      if (!layer.bbox) continue;
      const [x0, y0, x1, y1] = layer.bbox;
      if (x >= x0 && x < x1 && y >= y0 && y < y1) {
        return layer.id;
      }
    }
    return null;
  }

  if (!pack) {
    return (
      <div className="text-xs text-zinc-500">waiting for pack info…</div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3 text-xs text-zinc-400">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showOverlays}
            onChange={(e) => setShowOverlays(e.target.checked)}
          />
          Overlays
        </label>
        <span className="text-zinc-500">|</span>
        <span>{layers.length} layers</span>
        {selected && (
          <>
            <span className="text-zinc-500">|</span>
            <span className="text-accent-violet">selected: {selected}</span>
          </>
        )}
      </div>
      <canvas
        ref={canvasRef}
        className="w-full bg-ink-900 border border-ink-600 cursor-crosshair"
        onMouseMove={(e) => setHoverId(pickAtClient(e))}
        onMouseLeave={() => setHoverId(null)}
        onClick={(e) => setSelected(pickAtClient(e))}
      />
      {hoverId && (
        <div className="text-xs text-zinc-400">hover: {hoverId}</div>
      )}
    </div>
  );
}
