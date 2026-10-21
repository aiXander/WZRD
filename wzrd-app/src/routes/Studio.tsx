// Studio — where content is made, browsed and labelled (studio plan A2).
//
//   ┌ roots + folder tree │ folder grid (static thumbs, badges, job tiles) │ monitor + record ┐
//   └──────────────── style tray (the hub selection) · target folder ─────────────────────────┘
//
// The open folder is the target folder: renders started from here (or by
// an agent reading `studio_selection()`) land in it. Keys (only while no
// input has focus): arrows move (⇧ extends), space plays in the monitor,
// 0–9 score (− clears), T tag, N note, M move, D add to the open show’s
// active deck, F / Enter full-screen (ui/media/Lightbox over the folder),
// ⌫ delete media (to the trash for 30 min, Undo in the toast; records are
// kept; a multi-selection needs a second ⌫ within 2 s), Esc deselects.
// Every key is off while a modal / lightbox is open (show-setup G11).
// No telemetry subscriptions here — the only engine use is the monitor's
// `preview.file`.

import { useEffect, useRef, useState } from 'react';
import { Pane, Split } from '../ui/layoutSplit';
import { FolderTree } from '../components/studio/FolderTree';
import { MediaGrid, gridColumns } from '../components/studio/MediaGrid';
import { Monitor } from '../components/studio/Monitor';
import { RecordPanel } from '../components/studio/RecordPanel';
import { RenderPanel } from '../components/studio/RenderPanel';
import { StyleTray } from '../components/studio/StyleTray';
import { KeyPrompt, type PromptKind } from '../components/studio/KeyPrompt';
import { Lightbox } from '../ui/media/Lightbox';
import { modalOpen } from '../ui/media/modalState';
import { softDelete } from '../ui/media/trash';
import { toMediaRef } from '../ui/media/types';
import { Btn, Seg } from '../ui/primitives';
import { addToDeck } from '../state/decks';
import { useHub, useJobsFeed } from '../state/hub';
import {
  clickItem,
  deleteMedia,
  lastDir,
  loadSelection,
  moveFocus,
  openDir,
  refreshListing,
  setScore,
  targets,
  useStudio,
  visibleItems,
} from '../state/studio';

const POLL_MS = 2000;
const TAB_KEY = 'wzrd.studio.side';

type SideTab = 'record' | 'render';

/** Under the monitor: the focused item's record + labels, or the Render
 *  panel (R1). The choice is a per-machine convenience (localStorage). */
function SideTabs({ onPrompt }: { onPrompt: (k: PromptKind) => void }) {
  const [tab, setTab] = useState<SideTab>(() => {
    try {
      return localStorage.getItem(TAB_KEY) === 'render' ? 'render' : 'record';
    } catch {
      return 'record';
    }
  });
  const pick = (t: SideTab) => {
    setTab(t);
    try {
      localStorage.setItem(TAB_KEY, t);
    } catch {
      /* ignore */
    }
  };
  return (
    <div className="flex-1 min-h-0 flex flex-col gap-1.5">
      <Seg<SideTab>
        value={tab}
        onChange={pick}
        options={[
          { id: 'record', label: 'Record' },
          { id: 'render', label: 'Render' },
        ]}
      />
      {tab === 'record' ? (
        <RecordPanel onPrompt={onPrompt} />
      ) : (
        <section className="flex-1 min-h-0 border border-line rounded-lg bg-panel">
          <RenderPanel />
        </section>
      )}
    </div>
  );
}

export function Studio() {
  const hubState = useHub((s) => s.status?.state);
  const mediaRoot = useHub((s) => s.status?.media_root);
  const dir = useStudio((s) => s.dir);
  const [prompt, setPrompt] = useState<PromptKind | null>(null);
  const armedDelete = useRef<number>(0);
  useJobsFeed();

  // First open: the last folder, else the library home.
  useEffect(() => {
    if (!dir && mediaRoot) openDir(lastDir() ?? mediaRoot);
  }, [dir, mediaRoot]);

  // A1: poll the open folder every 2 s, only while Studio is mounted.
  useEffect(() => {
    if (!dir) return;
    const t = setInterval(refreshListing, POLL_MS);
    return () => clearInterval(t);
  }, [dir]);

  useEffect(() => {
    if (hubState === 'up') {
      loadSelection();
      refreshListing();
    }
  }, [hubState]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey || prompt || modalOpen()) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) {
        return;
      }
      const st = useStudio.getState();
      const cols = gridColumns();
      const digit = /^Digit([0-9])$/.exec(e.code)?.[1];
      if (digit && !e.shiftKey) {
        e.preventDefault();
        setScore(targets(), Number(digit));
        return;
      }
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          return moveFocus(-1, e.shiftKey);
        case 'ArrowRight':
          e.preventDefault();
          return moveFocus(1, e.shiftKey);
        case 'ArrowUp':
          e.preventDefault();
          return moveFocus(-cols, e.shiftKey);
        case 'ArrowDown':
          e.preventDefault();
          return moveFocus(cols, e.shiftKey);
        case ' ':
          e.preventDefault();
          st.setMonitor(st.monitor === st.focus ? null : st.focus);
          return;
        case '-':
          return void setScore(targets(), null);
        case 't':
        case 'T':
          e.preventDefault();
          return setPrompt('tag');
        case 'n':
        case 'N':
          e.preventDefault();
          return setPrompt('note');
        case 'd':
        case 'D':
          e.preventDefault();
          return void addToDeck(targets().filter((t) => t.kind === 'video'));
        case 'm':
        case 'M':
          e.preventDefault();
          return setPrompt('move');
        case 'f':
        case 'F':
        case 'Enter':
          if (!st.focus) return;
          e.preventDefault();
          useStudio.setState({ lightbox: st.focus });
          return;
        case 'Escape':
          useStudio.setState({ selected: [] });
          return;
        case 'Delete':
        case 'Backspace': {
          e.preventDefault();
          const items = targets();
          if (!items.length) return;
          // One item: immediate. Several: press again within 2 s.
          if (items.length > 1 && Date.now() - armedDelete.current > 2000) {
            armedDelete.current = Date.now();
            useStudio.setState({ deleteArmed: items.length });
            return;
          }
          armedDelete.current = 0;
          useStudio.setState({ deleteArmed: 0 });
          deleteMedia(items);
          return;
        }
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [prompt]);

  return (
    <div className="h-full flex flex-col p-2 gap-2 min-h-0">
      <Split id="studio.cols" dir="row" className="flex-1 min-h-0">
        <Pane id="tree" size="240px" min={180} label="Folders">
          <FolderTree />
        </Pane>
        <Pane id="grid" fill min={360} label="Folder">
          <MediaGrid />
        </Pane>
        <Pane id="side" size="400px" min={300} label="Monitor + record">
          <div className="h-full min-h-0 flex flex-col gap-2">
            <Monitor />
            <SideTabs onPrompt={setPrompt} />
          </div>
        </Pane>
      </Split>
      <StyleTray />
      {prompt && <KeyPrompt kind={prompt} onClose={() => setPrompt(null)} />}
      <StudioLightbox />
    </div>
  );
}

/** F / Enter / a tile's ⤢: the focused item full-screen, ← → over the
 *  folder's visible items (the cursor follows, so closing leaves it there). */
function StudioLightbox() {
  const path = useStudio((s) => s.lightbox);
  const listing = useStudio((s) => s.listing);
  const maxScore = useStudio((s) => s.maxScore);
  if (!path) return null;
  const items = visibleItems({ listing, maxScore }).map(toMediaRef);
  const index = items.findIndex((m) => m.path === path);
  const close = () => useStudio.setState({ lightbox: null });
  return (
    <Lightbox
      items={items}
      index={Math.max(0, index)}
      onIndex={(i) => {
        const p = items[i]?.path;
        if (!p) return;
        useStudio.setState({ lightbox: p });
        clickItem(p, {});
      }}
      onClose={close}
      actions={(ref) => (
        <Btn
          size="sm"
          tone="danger"
          onClick={() => {
            const i = items.findIndex((m) => m.path === ref.path);
            const next = items[i + 1] ?? items[i - 1];
            useStudio.setState((s) => ({ lightbox: next?.path ?? null, monitor: s.monitor === ref.path ? null : s.monitor }));
            softDelete([ref.path], refreshListing).then(refreshListing);
          }}
          title="Delete (to the trash — restorable for 30 min)"
        >
          Delete
        </Btn>
      )}
    />
  );
}
