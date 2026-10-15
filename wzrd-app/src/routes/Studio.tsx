// Studio — where content is made, browsed and labelled (studio plan A2).
//
//   ┌ roots + folder tree │ folder grid (static thumbs, badges, job tiles) │ monitor + record ┐
//   └──────────────── style tray (the hub selection) · target folder ─────────────────────────┘
//
// The open folder is the target folder: renders started from here (or by
// an agent reading `studio_selection()`) land in it. Keys (only while no
// input has focus): arrows move (⇧ extends), space plays in the monitor,
// 0–9 score (− clears), T tag, N note, M move, ⌫ delete media (records are
// kept; a multi-selection needs a second ⌫ within 2 s), Esc deselects.
// No telemetry subscriptions here — the only engine use is the monitor's
// `preview.file`.

import { useEffect, useRef, useState } from 'react';
import { Pane, Split } from '../ui/layoutSplit';
import { FolderTree } from '../components/studio/FolderTree';
import { MediaGrid, gridColumns } from '../components/studio/MediaGrid';
import { Monitor } from '../components/studio/Monitor';
import { RecordPanel } from '../components/studio/RecordPanel';
import { StyleTray } from '../components/studio/StyleTray';
import { KeyPrompt, type PromptKind } from '../components/studio/KeyPrompt';
import { useHub, useJobsFeed } from '../state/hub';
import {
  deleteMedia,
  lastDir,
  loadSelection,
  moveFocus,
  openDir,
  refreshListing,
  setScore,
  targets,
  useStudio,
} from '../state/studio';

const POLL_MS = 2000;

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
      if (e.metaKey || e.ctrlKey || e.altKey || prompt) return;
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
        case 'm':
        case 'M':
          e.preventDefault();
          return setPrompt('move');
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
            <RecordPanel onPrompt={setPrompt} />
          </div>
        </Pane>
      </Split>
      <StyleTray />
      {prompt && <KeyPrompt kind={prompt} onClose={() => setPrompt(null)} />}
    </div>
  );
}
