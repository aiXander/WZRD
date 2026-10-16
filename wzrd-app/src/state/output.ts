// The projector output switch (src-tauri/src/output.rs). Its own small store:
// the output window outlives every project switch.
//
// State is pushed by the shell as `output:state` / `output:displays`; the
// browser seat gets no Tauri events, so it re-reads while the popover is open.
// Flips + fit are NOT here — they live in the alignment document (engine-side,
// per project) and are written through `state/alignment.ts`.

import { create } from 'zustand';
import { call, IN_TAURI, listenEvent } from '../api/transport';

export type DisplayInfo = {
  index: number;
  name: string;
  /** Current mode, pixels — what the swapchain gets. */
  width: number;
  height: number;
  x: number;
  y: number;
  points_w: number;
  points_h: number;
  scale: number;
  refresh_hz: number | null;
  /** Native mode when the display reports one; ≠ width×height → macOS scales. */
  native: [number, number] | null;
  builtin: boolean;
  /** Holds the operator UI right now. */
  operator: boolean;
};

export type OutputTarget = 'off' | 'window' | 'display';

export type OutputState = {
  target: OutputTarget;
  display: string | null;
  last_display: string | null;
  hide_cursor: boolean;
  keep_awake: boolean;
  /** A display vanished under a live output. */
  lost: string | null;
};

type Store = {
  state: OutputState | null;
  displays: DisplayInfo[];
  error: string | null;
};

export const useOutput = create<Store>(() => ({ state: null, displays: [], error: null }));

const set = (p: Partial<Store>) => useOutput.setState(p);

async function run(p: Promise<OutputState>) {
  try {
    set({ state: await p, error: null });
  } catch (e) {
    set({ error: String(e) });
  }
}

export async function refreshOutput() {
  try {
    const [state, displays] = await Promise.all([
      call<OutputState>('output_state'),
      call<DisplayInfo[]>('output_displays'),
    ]);
    set({ state, displays });
  } catch (e) {
    set({ error: String(e) });
  }
}

export const outputOff = () => run(call<OutputState>('output_set', { target: 'off', display: null }));
export const outputWindow = () => run(call<OutputState>('output_set', { target: 'window', display: null }));
export const outputToDisplay = (name: string) =>
  run(call<OutputState>('output_set', { target: 'display', display: name }));
export const setOutputOptions = (o: { hideCursor?: boolean; keepAwake?: boolean }) =>
  run(call<OutputState>('output_options', { hideCursor: o.hideCursor ?? null, keepAwake: o.keepAwake ?? null }));
export const identifyDisplays = () => call<void>('output_identify').catch((e) => set({ error: String(e) }));

/** App-level, once: seed + follow the shell's pushes. */
export function startOutputSync() {
  void refreshOutput();
  if (!IN_TAURI) return;
  void listenEvent<OutputState>('output:state', (state) => set({ state }));
  void listenEvent<DisplayInfo[]>('output:displays', (displays) => set({ displays }));
}
