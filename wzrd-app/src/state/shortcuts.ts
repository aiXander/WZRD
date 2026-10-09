// Performance keyboard. Single keys, no modifiers for the things you do
// mid-set (a laptop on a dark stand, one hand on the trackpad). Ignored while
// typing in any field or the code editor.

import { useEffect } from 'react';
import {
  bakeDesignPins,
  controlLeg,
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
} from './actions';
import { flushScene } from './sceneCommit';
import { parseScene } from './scene';
import { useStore, type Route } from './store';

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

const ROUTE_KEYS: Record<string, Route> = { '1': 'prepare', '2': 'align', '3': 'perform', '4': 'debug' };

export function useShortcuts() {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
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
        }
        return;
      }
      if (typing(e)) return;
      if (e.key === 'Escape') {
        st.setShowHelp(false);
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
