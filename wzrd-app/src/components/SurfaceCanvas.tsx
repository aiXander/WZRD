// Surface canvas — the photo + mask overlays + named regions.
//
// The surface canvas is the primary visual on every page that has one (per
// `user_design_spec.md`): the live preview underneath, per-layer mask tints
// on top, readable region labels, and pixel-accurate hover/click picking.
//
// Performance notes:
//   - Each layer's tinted overlay is composited ONCE into an offscreen
//     canvas when its mask PNG loads, then blitted per repaint. (The old
//     code ran a full-canvas `source-in` fill per layer per preview frame,
//     which both wiped the canvas — only the last tint survived — and
//     burned the webview main thread at 15 fps.)
//   - Picking reads a downsampled per-layer alpha map (256 px wide), so
//     hover hit-testing is mask-accurate without holding full-res
//     ImageData for every layer.

import { useEffect, useMemo, useRef, useState } from 'react';
import { readMaskPng } from '../api/ipc';
import { useStore } from '../state/store';
import type { PackInfo, PackLayer } from '../api/ipc';

// One stable hue per layer index; same colour comes back across renders.
function hueForIndex(i: number): number {
  return (i * 47) % 360;
}

const PICK_W = 256;

type LayerAssets = {
  /** Full-res tinted overlay, pre-composited once. */
  overlay: HTMLCanvasElement;
  /** Downsampled alpha map for pixel picking. */
  pickData: Uint8ClampedArray;
  pickW: number;
  pickH: number;
};

function buildAssets(img: HTMLImageElement, w: number, h: number, hue: number): LayerAssets {
  const overlay = document.createElement('canvas');
  overlay.width = w;
  overlay.height = h;
  const octx = overlay.getContext('2d')!;
  octx.drawImage(img, 0, 0, w, h);
  octx.globalCompositeOperation = 'source-in';
  octx.fillStyle = `hsl(${hue}, 75%, 60%)`;
  octx.fillRect(0, 0, w, h);

  const pickH = Math.max(1, Math.round((h / w) * PICK_W));
  const pick = document.createElement('canvas');
  pick.width = PICK_W;
  pick.height = pickH;
  const pctx = pick.getContext('2d', { willReadFrequently: true })!;
  pctx.drawImage(img, 0, 0, PICK_W, pickH);
  const pickData = pctx.getImageData(0, 0, PICK_W, pickH).data;

  return { overlay, pickData, pickW: PICK_W, pickH };
}

/** Which bindings' selectors resolve to this layer? */
function bindingsForLayer(sceneJson: string, pack: PackInfo, layer: PackLayer): string[] {
  let scene: any;
  try {
    scene = JSON.parse(sceneJson);
  } catch {
    return [];
  }
  const groups = new Set(
    (pack.groups ?? [])
      .filter((g) => g.members.includes(layer.id))
      .map((g) => g.id)
  );
  const out: string[] = [];
  for (const b of scene.bindings ?? []) {
    const s = b.select ?? {};
    const hit =
      s.all === true ||
      s.id === layer.id ||
      (s.tag && layer.tags.includes(s.tag)) ||
      (s.group && groups.has(s.group));
    if (hit) out.push(b.id);
  }
  return out;
}

export function SurfaceCanvas() {
  const pack = useStore((s) => s.pack);
  const preview = useStore((s) => s.preview);
  const sceneJson = useStore((s) => s.sceneJson);
  const selectedLayer = useStore((s) => s.selectedLayerId);
  const setSelectedLayer = useStore((s) => s.setSelectedLayerId);
  const setSelectedBinding = useStore((s) => s.setSelectedBindingId);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const assetsRef = useRef<Record<string, LayerAssets>>({});
  const [assetsVersion, setAssetsVersion] = useState(0);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [showOverlays, setShowOverlays] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [previewImg, setPreviewImg] = useState<HTMLImageElement | null>(null);

  // Load every layer's mask PNG once (in parallel) and pre-composite its
  // tinted overlay + picking map.
  useEffect(() => {
    if (!pack) return;
    let cancelled = false;
    Promise.all(
      pack.layers.map(async (layer, i) => {
        if (assetsRef.current[layer.id]) return;
        try {
          const b64 = await readMaskPng(layer.mask_path);
          if (cancelled) return;
          const img = new Image();
          img.src = `data:image/png;base64,${b64}`;
          await img.decode().catch(() => {});
          if (cancelled) return;
          assetsRef.current[layer.id] = buildAssets(
            img,
            pack.width,
            pack.height,
            hueForIndex(i)
          );
          setAssetsVersion((v) => v + 1);
        } catch (e) {
          console.warn('mask load', layer.id, e);
        }
      })
    );
    return () => {
      cancelled = true;
    };
  }, [pack]);

  // Decode preview frames off the paint path — repaint only once decoded.
  useEffect(() => {
    if (!preview) return;
    let cancelled = false;
    const img = new Image();
    img.src = `data:image/jpeg;base64,${preview.data_b64}`;
    img
      .decode()
      .then(() => {
        if (!cancelled) setPreviewImg(img);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [preview]);

  // Repaint. All expensive work is pre-composited; this is a handful of
  // drawImage calls.
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
    ctx.fillStyle = '#08090b';
    ctx.fillRect(0, 0, pack.width, pack.height);

    if (previewImg) {
      ctx.drawImage(previewImg, 0, 0, pack.width, pack.height);
    }

    if (showOverlays) {
      pack.layers.forEach((layer) => {
        const a = assetsRef.current[layer.id];
        if (!a) return;
        const active = selectedLayer === layer.id || hoverId === layer.id;
        ctx.globalAlpha = active ? 0.55 : 0.22;
        ctx.drawImage(a.overlay, 0, 0);
      });
      ctx.globalAlpha = 1;
    }

    if (showLabels) {
      // Font size in pack-space so labels stay readable after CSS downscale.
      const fontPx = Math.max(14, Math.round(pack.width / 55));
      ctx.font = `${fontPx}px ui-monospace, monospace`;
      ctx.textBaseline = 'middle';
      ctx.lineWidth = Math.max(2, fontPx / 6);
      pack.layers.forEach((layer, i) => {
        if (!layer.centroid) return;
        const label = layer.label ?? layer.id;
        const [cx, cy] = layer.centroid;
        ctx.strokeStyle = 'rgba(0,0,0,0.85)';
        ctx.strokeText(label, cx + 4, cy);
        ctx.fillStyle =
          selectedLayer === layer.id || hoverId === layer.id
            ? '#ffffff'
            : `hsl(${hueForIndex(i)}, 70%, 75%)`;
        ctx.fillText(label, cx + 4, cy);
      });
    }
  }, [pack, previewImg, assetsVersion, selectedLayer, hoverId, showOverlays, showLabels]);

  const layers = useMemo(() => pack?.layers ?? [], [pack]);

  // Pixel-accurate picking against the downsampled alpha maps. Topmost
  // (highest z, then latest in pack order) wins.
  function pickAtClient(evt: React.MouseEvent<HTMLCanvasElement>): string | null {
    if (!pack) return null;
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    const u = (evt.clientX - rect.left) / rect.width;
    const v = (evt.clientY - rect.top) / rect.height;
    const ordered = [...layers].sort((a, b) => a.z - b.z);
    for (let i = ordered.length - 1; i >= 0; i--) {
      const layer = ordered[i];
      const a = assetsRef.current[layer.id];
      if (!a) continue;
      const px = Math.min(a.pickW - 1, Math.max(0, Math.floor(u * a.pickW)));
      const py = Math.min(a.pickH - 1, Math.max(0, Math.floor(v * a.pickH)));
      const alpha = a.pickData[(py * a.pickW + px) * 4 + 3];
      if (alpha > 32) return layer.id;
    }
    return null;
  }

  function onCanvasClick(evt: React.MouseEvent<HTMLCanvasElement>) {
    const layerId = pickAtClient(evt);
    setSelectedLayer(layerId);
    if (layerId && pack) {
      // Aim the inspector at the first binding whose selector covers this
      // region — "selecting a layer aims everything else" (design spec).
      const layer = pack.layers.find((l) => l.id === layerId);
      const targets = layer ? bindingsForLayer(sceneJson, pack, layer) : [];
      setSelectedBinding(targets[0] ?? null);
    } else {
      setSelectedBinding(null);
    }
  }

  const hoverBindings = useMemo(() => {
    if (!hoverId || !pack) return [];
    const layer = pack.layers.find((l) => l.id === hoverId);
    return layer ? bindingsForLayer(sceneJson, pack, layer) : [];
  }, [hoverId, pack, sceneJson]);

  if (!pack) {
    return <div className="text-xs text-zinc-500">waiting for pack info…</div>;
  }

  return (
    <div className="flex flex-col gap-2 h-full">
      <div className="flex items-center gap-3 text-xs text-zinc-400">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showOverlays}
            onChange={(e) => setShowOverlays(e.target.checked)}
          />
          Overlays
        </label>
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
          />
          Labels
        </label>
        <span className="text-zinc-500">|</span>
        <span>{layers.length} layers</span>
        {selectedLayer && (
          <>
            <span className="text-zinc-500">|</span>
            <span className="text-accent-violet">selected: {selectedLayer}</span>
          </>
        )}
      </div>
      <canvas
        ref={canvasRef}
        className="w-full bg-ink-900 border border-ink-600 cursor-crosshair rounded"
        onMouseMove={(e) => setHoverId(pickAtClient(e))}
        onMouseLeave={() => setHoverId(null)}
        onClick={onCanvasClick}
      />
      <div className="text-xs text-zinc-400 min-h-[1.25rem]">
        {hoverId ? (
          <>
            <span className="text-zinc-200">{hoverId}</span>
            {hoverBindings.length > 0 && (
              <span className="text-zinc-500">
                {' '}
                · bound by {hoverBindings.join(', ')}
              </span>
            )}
          </>
        ) : (
          <span className="text-zinc-600">
            hover a region to inspect · click to select
          </span>
        )}
      </div>
    </div>
  );
}
