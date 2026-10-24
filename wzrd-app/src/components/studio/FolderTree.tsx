// Studio's left column (A2): the library home + extra roots (L1), each a
// lazily expanded folder tree, "Index this folder" (S3 — a deep index so
// search sees the whole tree; opening a folder already indexes it), and
// "+ Add folder…" for a new root indexed in place (S2: never moved). The
// open folder is always revealed (ancestors expanded, row highlighted and
// scrolled into view) — the grid's breadcrumb opens folders too.

import { useEffect, useRef, useState } from 'react';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { addRoot, indexFolder, removeRoot, type Root } from '../../api/hub';
import { IN_TAURI } from '../../api/transport';
import { useHub } from '../../state/hub';
import { loadRoots, openDir, revealInTree, short, toggleTree, useStudio } from '../../state/studio';
import { useStore } from '../../state/store';
import { Btn } from '../../ui/primitives';

export function FolderTree() {
  const hubState = useHub((s) => s.status?.state);
  const roots = useStudio((s) => s.roots);
  const dir = useStudio((s) => s.dir);
  const reload = loadRoots;
  useEffect(() => {
    reload();
  }, [hubState]);
  useEffect(() => {
    if (dir && roots.length) revealInTree(dir);
  }, [dir, roots]);

  const add = async () => {
    let path: string | null = null;
    if (IN_TAURI) {
      const r = await openDialog({ directory: true, multiple: false, title: 'Index a folder in place' });
      path = typeof r === 'string' ? r : null;
    } else {
      path = window.prompt('Absolute folder path to index in place');
    }
    if (!path) return;
    try {
      await addRoot(path);
      await reload();
      openDir(path);
    } catch (e) {
      useStore.getState().toast(`Add folder failed: ${e}`, 'bad');
    }
  };

  return (
    <section className="h-full min-h-0 flex flex-col border border-line rounded-lg bg-panel">
      <header className="flex items-center gap-2 h-8 px-3 border-b border-line shrink-0">
        <div className="label">Library</div>
        <div className="flex-1" />
        <Btn size="xs" tone="ghost" onClick={add} disabled={hubState !== 'up'} title="Index another folder in place (e.g. ~/Visuals)">
          + Add folder…
        </Btn>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto py-1 text-[12px]">
        {roots.map((r) => (
          <RootRow key={r.path} root={r} onRemoved={reload} />
        ))}
      </div>
      <footer className="p-2 border-t border-line shrink-0">
        <IndexButton dir={dir} />
      </footer>
    </section>
  );
}

function RootRow({ root, onRemoved }: { root: Root; onRemoved: () => void }) {
  const name = root.home ? 'WZRD media' : root.path.split('/').pop() || root.path;
  return (
    <div className="group/root">
      <Node path={root.path} name={name} depth={0} title={root.path} />
      {!root.home && (
        <button
          className="hidden group-hover/root:block ml-6 text-[10px] text-fg3 hover:text-bad"
          onClick={() =>
            removeRoot(root.path)
              .then(onRemoved)
              .catch((e) => useStore.getState().toast(`${e}`, 'bad'))
          }
          title="Stop indexing this folder (files are not touched)"
        >
          remove from library
        </button>
      )}
    </div>
  );
}

function Node({ path, name, depth, title }: { path: string; name: string; depth: number; title?: string }) {
  const children = useStudio((s) => s.expanded[path]);
  const current = useStudio((s) => s.dir === path);
  const row = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (current) row.current?.scrollIntoView({ block: 'nearest' });
  }, [current]);
  return (
    <>
      <div
        ref={row}
        className={
          'flex items-center gap-1 h-6 pr-2 cursor-pointer whitespace-nowrap ' +
          (current ? 'bg-raised text-fg' : 'text-fg2 hover:bg-panel2 hover:text-fg')
        }
        style={{ paddingLeft: 6 + depth * 12 }}
        title={title ?? short(path)}
        onClick={() => openDir(path)}
      >
        <button
          className="w-4 text-fg3 hover:text-fg shrink-0"
          onClick={(e) => {
            e.stopPropagation();
            toggleTree(path);
          }}
        >
          {children ? '▾' : '▸'}
        </button>
        <span className="truncate">{name}</span>
      </div>
      {children?.map((c) => (
        <Node key={c.path} path={c.path} name={c.name} depth={depth + 1} />
      ))}
    </>
  );
}

function IndexButton({ dir }: { dir: string | null }) {
  const hubState = useHub((s) => s.status?.state);
  const [busy, setBusy] = useState(false);
  return (
    <Btn
      size="sm"
      className="w-full"
      disabled={!dir || hubState !== 'up' || busy}
      title="Index every media file below this folder (fingerprint, probe, thumbnail) so search sees it"
      onClick={async () => {
        if (!dir) return;
        setBusy(true);
        try {
          const r = await indexFolder(dir, true);
          useStore.getState().toast(`Indexing ${r.queued} of ${r.files} files below ${short(dir)}`, 'ok');
        } catch (e) {
          useStore.getState().toast(`Index failed: ${e}`, 'bad');
        } finally {
          setBusy(false);
        }
      }}
    >
      Index this folder
    </Btn>
  );
}
