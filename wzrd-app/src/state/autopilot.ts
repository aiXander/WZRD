// Auto-pilot (spec §13: "a chain of looks that keeps the surface alive while
// I'm away"). Walks the looks bank in order, one look every N bars. The next
// look is recalled into DESIGN half a bar *before* the boundary and TAKEn
// bar-quantized, so every fade starts exactly on the bar line.
//
// UI-driven for now: it runs while the app is open (it keeps going across
// rooms). The engine-side playlist in the roadmap supersedes it.

import { useEffect } from 'react';
import { create } from 'zustand';
import { recallLook } from './actions';
import { useStore } from './store';

type Auto = { on: boolean; bars: number; setOn: (v: boolean) => void; setBars: (n: number) => void };

export const useAutopilot = create<Auto>((set) => ({
  on: false,
  bars: 16,
  setOn: (v) => set({ on: v }),
  setBars: (n) => set({ bars: n }),
}));

export function useAutopilotDriver() {
  useEffect(() => {
    let lastCycle: number | null = null;
    return useStore.subscribe((s, prev) => {
      if (s.transport === prev.transport) return;
      const { on, bars } = useAutopilot.getState();
      const beats = s.transport?.live.beats;
      if (!on || beats == null) {
        lastCycle = null;
        return;
      }
      const span = 4 * bars;
      // Shifted by half a bar so the recall happens 2 beats early.
      const cycle = Math.floor((beats + 2) / span);
      if (lastCycle === null) {
        lastCycle = cycle;
        return;
      }
      if (cycle === lastCycle) return;
      lastCycle = cycle;
      const looks = s.looks;
      if (looks.length < 2 || s.deck?.promote !== 'idle') return;
      const i = looks.findIndex((l) => l.name === s.liveLook);
      recallLook(looks[(i + 1) % looks.length].name, true);
    });
  }, []);
}
