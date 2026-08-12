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
//   - Picking reads a downsampled per-layer mask map (256 px wide), so
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
  // Masks are opaque grayscale PNGs (mask in luminance, alpha = 255
  // everywhere), so move the mask into the alpha channel first — otherwise
  // the `source-in` tint keeps the whole canvas, not just the region.
  const od = octx.getImageData(0, 0, w, h);
  const d = od.data;
  for (let i = 0; i < d.length; i += 4) d[i + 3] = d[i];
  octx.putImageData(od, 0, 0);
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

/** Does this binding's selector cover this layer? */
function selectorCovers(select: any, pack: PackInfo, layer: PackLayer): boolean {
  const s = select ?? {};
  if (s.all === true) return true;
  if (s.id === layer.id) return true;
  if (s.tag && layer.tags.includes(s.tag)) return true;
  if (s.group) {
    const g = (pack.groups ?? []).find((grp) => grp.id === s.group);
    if (g && g.members.includes(layer.id)) return true;
  }
  return false;
}

/** Which bindings' selectors resolve to this layer? */
function bindingsForLayer(scene: any, pack: PackInfo, layer: PackLayer): string[] {
  const out: string[] = [];
  for (const b of scene?.bindings ?? []) {
    if (selectorCovers(b.select, pack, layer)) out.push(b.id);
  }
  return out;
}

/**
 * Inverse of `bindingsForLayer`: which layers does this binding light up?
 *
 * `pick` is deliberately ignored — the engine resolves a selector to its
 * whole member set and lets `pick` toggle which member *draws*
 * (`scene.rs::resolve_selector`), so the member set is what "this binding
 * owns these regions" should show.
 */
function layersForBinding(
  scene: any,
  pack: PackInfo | null,
  bindingId: string | null
): Set<string> {
  const out = new Set<string>();
  if (!bindingId || !pack) return out;
  const binding = (scene?.bindings ?? []).find((b: any) => b.id === bindingId);
  if (!binding) return out;
  for (const layer of pack.layers) {
    if (selectorCovers(binding.select, pack, layer)) out.add(layer.id);
  }
  return out;
}

export function SurfaceCanvas() {
  const pack = useStore((s) => s.pack);
  const preview = useStore((s) => s.preview);
  const sceneJson = useStore((s) => s.sceneJson);
  const selectedLayer = useStore((s) => s.selectedLayerId);
  const setSelectedLayer = useStore((s) => s.setSelectedLayerId);
  const selectedBinding = useStore((s) => s.selectedBindingId);
  const hoveredBinding = useStore((s) => s.hoveredBindingId);
  const setSelectedBinding = useStore((s) => s.setSelectedBindingId);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const assetsRef = useRef<Record<string, LayerAssets>>({});
  const [assetsVersion, setAssetsVersion] = useState(0);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [showOverlays, setShowOverlays] = useState(true);
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

  // Parsed once per scene edit — the selector resolvers below run on every
  // pointer move.
  const scene = useMemo(() => {
    try {
      return JSON.parse(sceneJson);
    } catch {
      return null;
    }
  }, [sceneJson]);

  // Two highlight tiers, fed from both directions (canvas pointer and
  // inspector pointer/selection) so hovering a binding row lights up exactly
  // the regions that binding drives.
  //   hot  — what the pointer is on *right now*
  //   warm — what is currently selected
  const hot = useMemo(() => {
    const s = layersForBinding(scene, pack, hoveredBinding);
    if (hoverId) s.add(hoverId);
    return s;
  }, [scene, pack, hoveredBinding, hoverId]);

  const warm = useMemo(() => {
    const s = layersForBinding(scene, pack, selectedBinding);
    if (selectedLayer) s.add(selectedLayer);
    return s;
  }, [scene, pack, selectedBinding, selectedLayer]);

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
      // Paint coldest-first so a highlighted region is never buried under an
      // overlapping neighbour's dim wash — the whole point of the highlight
      // is that it reads at a glance.
      const anyHighlight = hot.size > 0 || warm.size > 0;
      const tier = (layer: PackLayer) =>
        hot.has(layer.id) ? 2 : warm.has(layer.id) ? 1 : 0;
      const alpha = [anyHighlight ? 0.1 : 0.22, 0.45, 0.62];

      [0, 1, 2].forEach((t) => {
        pack.layers.forEach((layer) => {
          if (tier(layer) !== t) return;
          const a = assetsRef.current[layer.id];
          if (!a) return;
          ctx.globalAlpha = alpha[t];
          ctx.drawImage(a.overlay, 0, 0);
        });
      });
      ctx.globalAlpha = 1;
    }

    // Region names are shown on hover (in the status line below the canvas),
    // not painted at centroids — centroid labels landed in the wrong place for
    // thin/concave regions and cluttered the surface.
  }, [pack, previewImg, assetsVersion, hot, warm, showOverlays]);

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
      // Red channel, not alpha — grayscale masks are opaque (alpha = 255
      // everywhere); R carries the mask either way.
      const maskVal = a.pickData[(py * a.pickW + px) * 4];
      if (maskVal > 32) return layer.id;
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
      const targets = layer ? bindingsForLayer(scene, pack, layer) : [];
      setSelectedBinding(targets[0] ?? null);
    } else {
      setSelectedBinding(null);
    }
  }

  const hoverBindings = useMemo(() => {
    if (!hoverId || !pack) return [];
    const layer = pack.layers.find((l) => l.id === hoverId);
    return layer ? bindingsForLayer(scene, pack, layer) : [];
  }, [hoverId, pack, scene]);

  /** Human label when the pack has one, else the raw id. */
  function nameOf(layerId: string): string {
    const l = pack?.layers.find((x) => x.id === layerId);
    return l?.label && l.label !== l.id ? l.label : layerId;
  }

  // Regions lit by the binding row under the pointer — reported in the same
  // status line so the inspector→surface link is legible even for a selector
  // (tag / group / all) that lights up several at once.
  const bindingHoverNames = useMemo(
    () =>
      hoveredBinding && !hoverId
        ? [...layersForBinding(scene, pack, hoveredBinding)]
        : [],
    [hoveredBinding, hoverId, scene, pack]
  );

  if (!pack) {
    return <div className="text-xs text-zinc-500">waiting for pack info…</div>;
  }

  return (
    <div className="flex flex-col gap-2 h-full">
      {/* Single line, no wrap: a long selected-layer name must never push
          the canvas down (layout shift on click). */}
      <div className="flex items-center gap-3 text-xs text-zinc-400 whitespace-nowrap overflow-hidden">
        <label className="flex items-center gap-1 cursor-pointer shrink-0">
          <input
            type="checkbox"
            checked={showOverlays}
            onChange={(e) => setShowOverlays(e.target.checked)}
          />
          Overlays
        </label>
        <span className="text-zinc-500 shrink-0">|</span>
        <span className="shrink-0">{layers.length} layers</span>
        {selectedLayer && (
          <>
            <span className="text-zinc-500 shrink-0">|</span>
            <span
              className="text-accent-violet truncate min-w-0"
              title={selectedLayer}
            >
              selected: {nameOf(selectedLayer)}
            </span>
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
      <div className="text-xs text-zinc-400 min-h-[1.25rem] whitespace-nowrap overflow-hidden text-ellipsis">
        {hoverId ? (
          <>
            <span className="text-zinc-200">{nameOf(hoverId)}</span>
            {hoverBindings.length > 0 && (
              <span className="text-zinc-500">
                {' '}
                · bound by {hoverBindings.join(', ')}
              </span>
            )}
          </>
        ) : hoveredBinding ? (
          <>
            <span className="text-zinc-200">{hoveredBinding}</span>
            <span className="text-zinc-500">
              {' '}
              ·{' '}
              {bindingHoverNames.length === 0
                ? 'selects no region'
                : `drives ${bindingHoverNames.map(nameOf).join(', ')}`}
            </span>
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
