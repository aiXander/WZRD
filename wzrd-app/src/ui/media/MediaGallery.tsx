// A grid of media (show-setup M1): a folder (listed through api/hub.ts
// `listFolder`, polled every 2 s while mounted — the hub indexes what's new;
// hub down → the read-only library.db reader, and no pin / delete) or a
// fixed `items` list.
//
// - Job tiles first: every running job whose `dest` is this folder
//   (state/hub `useJobsFeed` held while mounted with a `dir`).
// - Click = select (`onSelect`), double-click / ⤢ = Lightbox over the
//   gallery's items (`compareWith` passed through), ★ = NamePrompt →
//   `pinAsset(project, name, path)` (only with `project`; ★ chips show which
//   asset names point at each file), ✕ = soft delete + an Undo toast.
// - `extraActions(ref)` renders in the lightbox's top bar.

import { useEffect, useMemo, useState } from 'react';
import { listFolder, type FolderListing, type LibraryItem } from '../../api/hub';
import { pinAsset } from '../../api/project';
import { hubUp, useHub, useJobsFeed } from '../../state/hub';
import { DEFAULT_ROLES, pinsByPath, refreshProjectInfo, useProjectInfo } from '../../state/projectAssets';
import { useStore } from '../../state/store';
import { Btn } from '../primitives';
import { JobTile } from './JobTile';
import { Lightbox } from './Lightbox';
import { MediaTile } from './MediaTile';
import { NamePrompt } from './NamePrompt';
import { softDelete } from './trash';
import { baseName, sameDir, toMediaRef, type MediaRef } from './types';

const POLL_MS = 2000;

export function MediaGallery({
  dir,
  items: fixed,
  filter,
  sort = 'newest',
  project,
  roles,
  selectedPath,
  onSelect,
  onPinned,
  minTile = 150,
  emptyText,
  compareWith,
  allowDelete = true,
  extraActions,
  header,
}: {
  dir?: string | null;
  items?: MediaRef[];
  filter?: (it: LibraryItem) => boolean;
  sort?: 'newest' | 'name';
  project?: string | null;
  roles?: string[];
  selectedPath?: string | null;
  onSelect?: (ref: MediaRef) => void;
  onPinned?: (name: string, ref: MediaRef) => void;
  minTile?: number;
  emptyText?: React.ReactNode;
  compareWith?: string | null;
  allowDelete?: boolean;
  extraActions?: (ref: MediaRef) => React.ReactNode;
  header?: React.ReactNode;
}) {
  const up = useHub((s) => s.status?.state === 'up');
  const jobs = useHub((s) => s.jobs);
  const [listing, setListing] = useState<FolderListing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  const [pinFor, setPinFor] = useState<{ ref: MediaRef; anchor: HTMLElement } | null>(null);
  const { info } = useProjectInfo(project ?? null);
  const pins = useMemo(() => pinsByPath(info), [info]);
  const folder = fixed ? null : dir ?? null;
  useJobsFeed(!!folder);

  const reload = async () => {
    if (!folder) return;
    try {
      const l = await listFolder(folder, hubUp());
      setListing((prev) => (prev && sameDir(prev.dir, l.dir) && JSON.stringify(prev) === JSON.stringify(l) ? prev : l));
      setError(null);
    } catch (e) {
      // A show folder nothing has written to yet (segmentation/ before the
      // first render) is just empty, not an error.
      const msg = String(e);
      if (/No such file|not a folder/i.test(msg)) {
        setListing({ dir: folder, root: null, dirs: [], items: [], pending: [] });
        setError(null);
      } else setError(msg);
    }
  };

  useEffect(() => {
    setListing(null);
    setError(null);
    if (!folder) return;
    reload();
    const t = setInterval(reload, POLL_MS);
    return () => clearInterval(t);
  }, [folder]);

  const items: MediaRef[] = useMemo(() => {
    if (fixed) return fixed;
    let its = (listing?.items ?? []).filter((i) => !i.missing);
    if (filter) its = its.filter(filter);
    its = [...its].sort((a, b) =>
      sort === 'name' ? a.name.localeCompare(b.name) : (b.mtime ?? 0) - (a.mtime ?? 0) || a.name.localeCompare(b.name)
    );
    return its.map(toMediaRef);
  }, [fixed, listing, filter, sort]);

  const running = folder ? jobs.filter((j) => j.status === 'running' && sameDir(j.dest, folder)) : [];
  const canWrite = up;
  const suggestions = roles ?? (info ? Object.keys(info.roles ?? {}) : DEFAULT_ROLES);

  const del = async (ref: MediaRef) => {
    if (await softDelete([ref.path], reload)) reload();
  };

  const pin = async (name: string, ref: MediaRef) => {
    if (!project) return;
    try {
      await pinAsset(project, name, ref.path);
      await refreshProjectInfo(project);
      useStore.getState().toast(`Pinned ${baseName(ref.path)} as ${name}`, 'ok');
      onPinned?.(name, ref);
    } catch (e) {
      useStore.getState().toast(`Pin failed: ${e}`, 'bad');
    }
  };

  const lightboxActions = (ref: MediaRef) => (
    <>
      {extraActions?.(ref)}
      {project && canWrite && (
        <Btn size="sm" onClick={(e) => setPinFor({ ref, anchor: e.currentTarget as HTMLElement })} title="Pin to the show as a named asset">
          ★ Pin as…
        </Btn>
      )}
      {allowDelete && canWrite && (
        <Btn size="sm" tone="danger" onClick={() => del(ref)} title="Delete (to the trash — restorable for 30 min)">
          Delete
        </Btn>
      )}
    </>
  );

  return (
    <div className="flex flex-col min-h-0 gap-2">
      {header}
      {error && <div className="text-[11px] text-bad">{error}</div>}
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${minTile}px, 1fr))` }}>
        {running.map((j) => (
          <JobTile key={j.job} job={j} />
        ))}
        {items.map((ref, i) => (
          <MediaTile
            key={ref.path}
            media={ref}
            selected={!!selectedPath && ref.path === selectedPath}
            onClick={onSelect ? () => onSelect(ref) : () => setOpen(i)}
            onOpen={() => setOpen(i)}
            pins={pins[ref.path]}
            onPin={project && canWrite ? (anchor) => setPinFor({ ref, anchor }) : undefined}
            onDelete={allowDelete && canWrite ? () => del(ref) : undefined}
          />
        ))}
        {listing?.pending.map((p) => (
          <div
            key={p}
            className="aspect-video rounded-md border border-dashed border-line2 flex items-center justify-center text-[10px] text-fg3 px-2 text-center break-all"
            title="Being indexed (fingerprint, probe, thumbnail)"
          >
            {baseName(p)}
          </div>
        ))}
      </div>
      {!items.length && !running.length && !listing?.pending.length && (fixed || listing) && (
        <div className="text-[12px] text-fg3 p-4 text-center">{emptyText ?? 'Nothing here yet.'}</div>
      )}
      {open != null && items.length > 0 && (
        <Lightbox
          items={items}
          index={open}
          onIndex={(i) => {
            setOpen(i);
            if (onSelect && items[i]) onSelect(items[i]);
          }}
          onClose={() => setOpen(null)}
          compareWith={compareWith}
          actions={lightboxActions}
        />
      )}
      {pinFor && (
        <NamePrompt
          anchor={pinFor.anchor}
          suggestions={suggestions}
          initial={pins[pinFor.ref.path]?.[0] ?? (roles?.length === 1 ? roles[0] : '')}
          title={`Pin ${baseName(pinFor.ref.path)} as…`}
          onSubmit={(name) => pin(name, pinFor.ref)}
          onClose={() => setPinFor(null)}
        />
      )}
    </div>
  );
}
