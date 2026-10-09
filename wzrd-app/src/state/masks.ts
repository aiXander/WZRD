// Shared mask assets, loaded once per pack and used by every surface view:
// the tinted full-res overlay (surface map), a downsampled pick map (hover
// hit-testing), and a small silhouette thumbnail (layer rows, region chips).
// Also the pack's surface photo — "the surface map is primary" (spec §1).

import { useEffect, useState } from 'react';
import { readMaskPng, type PackInfo } from '../api/ipc';
import { hueForIndex } from './scene';

export type MaskAsset = {
  overlay: HTMLCanvasElement;
  pickData: Uint8ClampedArray;
  pickW: number;
  pickH: number;
  /** Silhouette PNG data URL (white on transparent) for small chips. */
  thumb: string;
  hue: number;
};

const PICK_W = 256;
const THUMB_W = 96;

let packKey: string | null = null;
let assets: Record<string, MaskAsset> = {};
let surface: HTMLImageElement | null = null;
let version = 0;
const subs = new Set<() => void>();
const bump = () => {
  version++;
  subs.forEach((f) => f());
};

function build(img: HTMLImageElement, w: number, h: number, hue: number): MaskAsset {
  const overlay = document.createElement('canvas');
  overlay.width = w;
  overlay.height = h;
  const octx = overlay.getContext('2d')!;
  octx.drawImage(img, 0, 0, w, h);
  // Masks are opaque grayscale: move luminance into alpha before tinting.
  const od = octx.getImageData(0, 0, w, h);
  const d = od.data;
  for (let i = 0; i < d.length; i += 4) d[i + 3] = d[i];
  octx.putImageData(od, 0, 0);
  octx.globalCompositeOperation = 'source-in';
  octx.fillStyle = `hsl(${hue}, 85%, 62%)`;
  octx.fillRect(0, 0, w, h);

  const pickH = Math.max(1, Math.round((h / w) * PICK_W));
  const pick = document.createElement('canvas');
  pick.width = PICK_W;
  pick.height = pickH;
  const pctx = pick.getContext('2d', { willReadFrequently: true })!;
  pctx.drawImage(img, 0, 0, PICK_W, pickH);
  const pickData = pctx.getImageData(0, 0, PICK_W, pickH).data;

  const th = Math.max(1, Math.round((h / w) * THUMB_W));
  const t = document.createElement('canvas');
  t.width = THUMB_W;
  t.height = th;
  const tctx = t.getContext('2d')!;
  tctx.drawImage(img, 0, 0, THUMB_W, th);
  const td = tctx.getImageData(0, 0, THUMB_W, th);
  for (let i = 0; i < td.data.length; i += 4) {
    const a = td.data[i];
    td.data[i] = td.data[i + 1] = td.data[i + 2] = 255;
    td.data[i + 3] = a;
  }
  tctx.putImageData(td, 0, 0);

  return { overlay, pickData, pickW: PICK_W, pickH, thumb: t.toDataURL(), hue };
}

export function ensureMasks(pack: PackInfo | null) {
  if (!pack) return;
  const key = `${pack.pack_dir}|${pack.layers.map((l) => l.mask_path).join(',')}`;
  if (key === packKey) return;
  packKey = key;
  assets = {};
  surface = null;
  bump();
  pack.layers.forEach(async (layer, i) => {
    try {
      const b64 = await readMaskPng(layer.mask_path);
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode().catch(() => {});
      if (packKey !== key) return;
      assets[layer.id] = build(img, pack.width, pack.height, hueForIndex(i));
      bump();
    } catch (e) {
      console.warn('mask load', layer.id, e);
    }
  });
  // The photographed surface (pack.json `surface`, conventionally surface.png).
  readMaskPng('surface.png')
    .then(async (b64) => {
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode();
      if (packKey !== key) return;
      surface = img;
      bump();
    })
    .catch(() => {});
}

export function getMask(id: string): MaskAsset | undefined {
  return assets[id];
}
export function getSurface(): HTMLImageElement | null {
  return surface;
}

/** Re-render on mask/surface arrival. */
export function useMaskVersion(): number {
  const [, set] = useState(version);
  useEffect(() => {
    const f = () => set(version);
    subs.add(f);
    return () => {
      subs.delete(f);
    };
  }, []);
  return version;
}

/** Topmost region under a uv point. */
export function pickRegion(pack: PackInfo, u: number, v: number): string | null {
  const ordered = [...pack.layers].sort((a, b) => a.z - b.z);
  for (let i = ordered.length - 1; i >= 0; i--) {
    const a = assets[ordered[i].id];
    if (!a) continue;
    const px = Math.min(a.pickW - 1, Math.max(0, Math.floor(u * a.pickW)));
    const py = Math.min(a.pickH - 1, Math.max(0, Math.floor(v * a.pickH)));
    if (a.pickData[(py * a.pickW + px) * 4] > 32) return ordered[i].id;
  }
  return null;
}
