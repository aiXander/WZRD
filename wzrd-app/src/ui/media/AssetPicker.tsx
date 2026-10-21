// Pick a media input for a show (show-setup M1): a compact select-like
// button (thumb + asset name / file name) opening a popover with
//   · Named assets — this show's `assets.json`, the `suggest` names first
//     (a suggested name nobody pinned yet, or an asset whose file is gone,
//     shows greyed),
//   · Recent in this show — the newest 24 files of the right kind across the
//     show's working folders (images: captures/ surfaces/ segmentation/
//     styles/; videos: generated/ video/ — paths from ProjectInfo.layout),
//   · Choose file… — the native dialog (hidden in the browser seat).
// `value` is an absolute path; picking a named asset gives its current path.

import { useEffect, useMemo, useRef, useState } from 'react';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { listFolder, mediaUrl, type LibraryItem } from '../../api/hub';
import type { Asset } from '../../api/project';
import { IN_TAURI } from '../../api/transport';
import { hubUp } from '../../state/hub';
import { pinsByPath, useProjectInfo } from '../../state/projectAssets';
import { Popover } from '../Popover';
import { baseName, isImagePath, isVideoPath, kindOf, thumbOf, toMediaRef, type MediaRef } from './types';

const RECENT_FOLDERS = { image: ['captures', 'surfaces', 'segmentation', 'styles'], video: ['generated', 'video'] };
const IMAGE_FILTER = ['png', 'jpg', 'jpeg', 'webp', 'tif', 'tiff', 'bmp', 'gif'];
const VIDEO_FILTER = ['mp4', 'mov', 'webm', 'm4v'];

export function AssetPicker({
  project,
  value,
  onChange,
  kind = 'image',
  suggest = [],
  placeholder = 'Choose…',
  allowClear = true,
  compact = false,
}: {
  project: string;
  value: string | null;
  onChange: (path: string | null) => void;
  kind?: 'image' | 'video';
  suggest?: string[];
  placeholder?: string;
  allowClear?: boolean;
  compact?: boolean;
}) {
  const { info } = useProjectInfo(project);
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement | null>(null);
  const pins = useMemo(() => pinsByPath(info), [info]);
  const valueName = value ? pins[value]?.[0] : undefined;
  const valueRef: MediaRef | null = value ? { path: value, kind: isVideoPath(value) ? 'video' : isImagePath(value) ? 'image' : null } : null;

  return (
    <>
      <button
        ref={btn}
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={value ?? placeholder}
        className={
          'flex items-center gap-2 w-full min-w-0 rounded border text-left bg-bg ' +
          (compact ? 'h-7 px-1 ' : 'h-10 px-1.5 ') +
          (open ? 'border-deck/60' : 'border-line hover:border-line2')
        }
      >
        <Thumb media={valueRef} size={compact ? 'sm' : 'md'} />
        <span className="flex-1 min-w-0 leading-tight">
          {value ? (
            <>
              <span className="block text-[12px] text-fg truncate">{valueName ? `★ ${valueName}` : baseName(value)}</span>
              {!compact && valueName && <span className="block text-[10px] text-fg3 truncate">{baseName(value)}</span>}
            </>
          ) : (
            <span className="text-[12px] text-fg3">{placeholder}</span>
          )}
        </span>
        <span className="text-[10px] text-fg3 shrink-0 pr-0.5">▾</span>
      </button>
      {open && (
        <PickerPopover
          anchor={btn.current}
          project={project}
          kind={kind}
          suggest={suggest}
          value={value}
          allowClear={allowClear}
          assets={info?.assets ?? {}}
          layout={info?.layout ?? {}}
          onPick={(p) => {
            onChange(p);
            setOpen(false);
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

function PickerPopover({
  anchor,
  project,
  kind,
  suggest,
  value,
  allowClear,
  assets,
  layout,
  onPick,
  onClose,
}: {
  anchor: HTMLElement | null;
  project: string;
  kind: 'image' | 'video';
  suggest: string[];
  value: string | null;
  allowClear: boolean;
  assets: Record<string, Asset>;
  layout: Record<string, { path: string; what: string }>;
  onPick: (p: string | null) => void;
  onClose: () => void;
}) {
  const [recent, setRecent] = useState<LibraryItem[] | null>(null);
  const folders = RECENT_FOLDERS[kind].map((f) => layout[f]?.path).filter((p): p is string => !!p);
  const foldersKey = folders.join('|');

  useEffect(() => {
    let live = true;
    if (!folders.length) {
      setRecent([]);
      return;
    }
    Promise.all(folders.map((d) => listFolder(d, hubUp()).then((l) => l.items).catch(() => [] as LibraryItem[])))
      .then((lists) => {
        if (!live) return;
        const all = lists
          .flat()
          .filter((i) => !i.missing && (i.kind ? i.kind === kind : kind === 'image' ? isImagePath(i.path) : isVideoPath(i.path)));
        all.sort((a, b) => (b.mtime ?? 0) - (a.mtime ?? 0));
        setRecent(all.slice(0, 24));
      });
    return () => {
      live = false;
    };
  }, [project, kind, foldersKey]);

  // Named assets: suggested names first (in the caller's order), then the rest A–Z.
  const rows = useMemo(() => {
    const names = [...suggest, ...Object.keys(assets).sort().filter((n) => !suggest.includes(n))];
    return names
      .map((name) => ({ name, asset: assets[name] as Asset | undefined }))
      .filter(({ asset }) => !asset || !asset.kind || asset.kind === kind || kindOf({ path: asset.path }) === kind);
  }, [assets, suggest, kind]);

  const choose = async () => {
    const r = await openDialog({
      multiple: false,
      directory: false,
      title: kind === 'image' ? 'Choose an image' : 'Choose a video',
      filters: [{ name: kind === 'image' ? 'Images' : 'Videos', extensions: kind === 'image' ? IMAGE_FILTER : VIDEO_FILTER }],
    });
    if (typeof r === 'string') onPick(r);
  };

  return (
    <Popover anchor={anchor} onClose={onClose} width={340}>
      <div className="max-h-[60vh] overflow-y-auto -mx-1 px-1 flex flex-col gap-2">
        <section>
          <div className="label mb-1">Named assets</div>
          {rows.length === 0 ? (
            <div className="text-[11px] text-fg3">None pinned in this show yet.</div>
          ) : (
            <div className="flex flex-col">
              {rows.map(({ name, asset }) => {
                const usable = !!asset && asset.exists !== false;
                return (
                  <button
                    key={name}
                    disabled={!usable}
                    onClick={() => asset && onPick(asset.path)}
                    title={asset ? (usable ? asset.path : `Missing: ${asset.path}`) : 'Not pinned in this show yet'}
                    className={
                      'flex items-center gap-2 px-1 py-0.5 rounded text-left min-w-0 ' +
                      (!usable ? 'opacity-40 cursor-default ' : 'hover:bg-panel2 ') +
                      (asset && asset.path === value ? 'bg-deck/15' : '')
                    }
                  >
                    <Thumb media={asset && usable ? { path: asset.path, kind: asset.kind } : null} size="sm" />
                    <span className="text-[12px] font-mono text-fg truncate">{name}</span>
                    <span className="flex-1" />
                    <span className="text-[10px] text-fg3 truncate max-w-[45%]">
                      {asset ? (usable ? baseName(asset.path) : 'missing') : 'not pinned'}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </section>
        <section>
          <div className="label mb-1">Recent in this show</div>
          {recent === null ? (
            <div className="text-[11px] text-fg3">Loading…</div>
          ) : recent.length === 0 ? (
            <div className="text-[11px] text-fg3">Nothing in this show's folders yet.</div>
          ) : (
            <div className="grid grid-cols-4 gap-1">
              {recent.map((it) => (
                <button
                  key={it.path}
                  onClick={() => onPick(it.path)}
                  title={it.path}
                  className={
                    'rounded border overflow-hidden bg-black aspect-video ' +
                    (it.path === value ? 'border-deck ring-1 ring-deck' : 'border-line hover:border-line2')
                  }
                >
                  <ThumbImg media={toMediaRef(it)} />
                </button>
              ))}
            </div>
          )}
        </section>
        <div className="flex items-center gap-1.5 pt-1 border-t border-line">
          {IN_TAURI && (
            <button className="text-[11px] text-fg2 hover:text-fg px-1 py-0.5 rounded hover:bg-panel2" onClick={choose}>
              Choose file…
            </button>
          )}
          <span className="flex-1" />
          {allowClear && value && (
            <button className="text-[11px] text-fg3 hover:text-bad px-1 py-0.5 rounded" onClick={() => onPick(null)}>
              Clear
            </button>
          )}
        </div>
      </div>
    </Popover>
  );
}

function ThumbImg({ media }: { media: MediaRef }) {
  const [broken, setBroken] = useState(false);
  const t = thumbOf(media);
  if (!t || broken) return <span className="text-[10px] text-fg3">{kindOf(media) === 'video' ? '▶' : '—'}</span>;
  return (
    <img
      src={mediaUrl(t)}
      loading="lazy"
      draggable={false}
      onError={() => setBroken(true)}
      className="w-full h-full object-contain"
    />
  );
}

function Thumb({ media, size }: { media: MediaRef | null; size: 'sm' | 'md' }) {
  return (
    <span
      className={
        'rounded bg-black border border-line shrink-0 overflow-hidden flex items-center justify-center ' +
        (size === 'sm' ? 'w-8 h-5' : 'w-12 h-8')
      }
    >
      {media ? <ThumbImg media={media} /> : <span className="text-[9px] text-fg4">—</span>}
    </span>
  );
}
