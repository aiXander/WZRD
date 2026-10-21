// The one media handle every ui/media component takes: a library item when
// we have one (thumb, size, labels), else just a path.

import type { LibraryItem } from '../../api/hub';

export type MediaRef = {
  path: string;
  name?: string;
  kind?: 'image' | 'video' | null;
  thumb?: string | null;
  w?: number | null;
  h?: number | null;
  item?: LibraryItem;
};

export function toMediaRef(item: LibraryItem): MediaRef {
  return { path: item.path, name: item.name, kind: item.kind, thumb: item.thumb, w: item.w, h: item.h, item };
}

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp|tiff?|avif|heic)$/i;
const VIDEO_EXT = /\.(mp4|mov|webm|m4v|mkv|avi)$/i;

export function isImagePath(p: string): boolean {
  return IMAGE_EXT.test(p);
}

export function isVideoPath(p: string): boolean {
  return VIDEO_EXT.test(p);
}

export const baseName = (p: string) => p.replace(/\/+$/, '').split('/').pop() ?? p;

/** The kind of a ref: its own, else by extension. */
export function kindOf(m: MediaRef): 'image' | 'video' | null {
  if (m.kind) return m.kind;
  if (isImagePath(m.path)) return 'image';
  if (isVideoPath(m.path)) return 'video';
  return null;
}

/** A static picture for a ref: its thumb, else the file itself if it's an image. */
export function thumbOf(m: MediaRef): string | null {
  if (m.thumb) return m.thumb;
  return kindOf(m) === 'image' ? m.path : null;
}

export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

/** Same folder, regardless of a trailing slash. */
export const sameDir = (a: string | null | undefined, b: string | null | undefined) =>
  !!a && !!b && a.replace(/\/+$/, '') === b.replace(/\/+$/, '');
