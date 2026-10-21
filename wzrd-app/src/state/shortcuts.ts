// Performance keyboard. Single keys, no modifiers for the things you do
// mid-set (a laptop on a dark stand, one hand on the trackpad). Ignored while
// typing in any field or the code editor.

import { useEffect } from 'react';
import {
  bakeDesignPins,
  clearSelection,
  controlLeg,
  deleteLayerOnHands,
  flash,
  toggleFreeze,
  mixOf,
  pullLive,
  recallLook,
  resyncBeat,
  setDeck,
  setOpacity,
  take,
  tapTempo,
  toggleBlackout,
  toggleMute,
  toggleSolo,
  undoDeleteLayer,
} from './actions';
import { flushScene } from './sceneCommit';
import { parseScene } from './scene';
import { useStore, type Route } from './store';
import { modalOpen } from '../ui/media/modalState';

function typing(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  if (t.closest('.monaco-editor')) return true;
  const tag = t.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable;
}

function legBindings(): string[] {
  const st = useStore.getState();
  const json = controlLeg() === 'live' ? st.liveSceneJson || st.sceneJson : st.sceneJson;
  return (parseScene(json)?.bindings ?? []).map((b) => b.id);
}

// ⌘0 Studio · ⌘1 Prepare · ⌘2 Perform = the tab bar (Studio got ⌘0 so the
// show rooms keep their keys); ⌘3/⌘4 = the ⚙ menu's rooms.
const ROUTE_KEYS: Record<string, Route> = {
  '0': 'studio',
  '1': 'prepare',
  '2': 'perform',
  '3': 'align',
  '4': 'debug',
};

export function useShortcuts() {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // A modal / lightbox owns the keyboard (show-setup G11): digits there
      // must not select layers, T must not TAKE, ⌘1 must not switch rooms
      // under a dialog.
      if (modalOpen()) return;
      const st = useStore.getState();
      if (e.metaKey || e.ctrlKey) {
        if (ROUTE_KEYS[e.key]) {
          e.preventDefault();
          st.setRoute(ROUTE_KEYS[e.key]);
        } else if (e.key === 's' && !typing(e)) {
          e.preventDefault();
          bakeDesignPins();
          flushScene();
          st.toast('Design saved to scene.json', 'ok');
        } else if ((e.key === 'z' || e.key === 'Z') && !e.shiftKey && !typing(e)) {
          // ⌘Z = undo the last layer delete (the only undoable act so far).
          if (undoDeleteLayer()) e.preventDefault();
        }
        return;
      }
      if (typing(e)) return;
      if (e.key === 'Escape') {
        // Help first; otherwise Esc deselects in Prepare (Perform's mixer
        // always keeps a layer selected, so there it would just jump).
        if (st.showHelp) st.setShowHelp(false);
        else if (st.route === 'prepare') clearSelection();
        return;
      }
      if (e.key === '?') {
        st.setShowHelp(!st.showHelp);
        return;
      }
      if (e.key === 'Tab') {
        e.preventDefault();
        tapTempo();
        return;
      }
      if (e.key === '`') return resyncBeat();
      // ⇧ + digit → cue look N (⇧⌥ → GO). Uses e.code: ⇧1 types "!".
      const digit = /^Digit([1-9])$/.exec(e.code)?.[1];
      if (digit && e.shiftKey) {
        const look = st.looks[Number(digit) - 1];
        if (look) recallLook(look.name, e.altKey);
        return;
      }
      if (st.route === 'align') return; // arrows/keys belong to the warp canvas
      // Studio's single keys (score digits, T tag, M move, ⌫ delete, arrows)
      // are its own (routes/Studio.tsx) — never TAKE / mute / layer delete.
      if (st.route === 'studio') return;
      const ids = legBindings();
      const sel = st.selectedBindingId;
      const idx = sel ? ids.indexOf(sel) : -1;
      switch (e.key) {
        case 'd':
        case 'D':
          return void setDeck('design');
        case 'l':
        case 'L':
          return void setDeck('live');
        case 't':
          return void take();
        case 'T':
          return void take({ cut: true });
        case 'p':
        case 'P':
          return void pullLive();
        case 'b':
        case 'B':
          return toggleBlackout();
        case 'f':
        case 'F':
          if (!e.repeat) flash(true);
          return;
        case 'z':
        case 'Z':
          return toggleFreeze();
        case 'm':
        case 'M':
          if (sel) toggleMute(sel);
          return;
        case 's':
        case 'S':
          if (sel) toggleSolo(sel, !e.shiftKey);
          return;
        case '[':
        case ']':
          if (sel) setOpacity(sel, Math.min(1, Math.max(0, mixOf(sel).opacity + (e.key === ']' ? 0.1 : -0.1))));
          return;
        case 'Delete':
        case 'Backspace':
          // On LIVE this moves hands to DESIGN instead (actions.ts).
          if (sel) {
            e.preventDefault();
            deleteLayerOnHands(sel);
          }
          return;
        case 'ArrowUp':
        case 'ArrowDown': {
          if (!ids.length) return;
          e.preventDefault();
          const n = idx < 0 ? 0 : (idx + (e.key === 'ArrowDown' ? 1 : -1) + ids.length) % ids.length;
          st.setSelectedBindingId(ids[n]);
          return;
        }
      }
      if (digit && ids[Number(digit) - 1]) st.setSelectedBindingId(ids[Number(digit) - 1]);
    }
    function onKeyUp(e: KeyboardEvent) {
      // Not gated by modalOpen(): a flash held when a modal opened must still release.
      if (e.key === 'f' || e.key === 'F') flash(false);
    }
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);
}
