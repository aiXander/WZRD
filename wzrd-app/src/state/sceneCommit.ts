// Debounced scene-commit path shared by every widget that mutates
// scene.json (binding inspector, driver rack, surface canvas).
//
// Semantics:
//   - The store's `sceneJson` updates immediately (optimistic — the UI never
//     waits on the engine).
//   - The engine push (`scene.load`, full plan rebuild) trails by ~150 ms so
//     a slider drag or rapid keystrokes collapse into one rebuild.
//   - The disk write trails by ~800 ms; the engine-side reload dedupe skips
//     the file-watcher echo of our own write.

import { sceneLoad, writeSceneFile } from '../api/ipc';
import { useStore } from './store';

let pushTimer: number | null = null;
let persistTimer: number | null = null;
let pending: string | null = null;

const PUSH_DEBOUNCE_MS = 150;
const PERSIST_DEBOUNCE_MS = 800;

export function commitSceneText(text: string) {
  const st = useStore.getState();
  st.setSceneJson(text);
  st.setSceneDirty(true);
  pending = text;

  if (pushTimer != null) window.clearTimeout(pushTimer);
  pushTimer = window.setTimeout(() => {
    pushTimer = null;
    if (pending == null) return;
    sceneLoad(pending).catch((e) => console.error('scene apply:', e));
  }, PUSH_DEBOUNCE_MS);

  if (persistTimer != null) window.clearTimeout(persistTimer);
  persistTimer = window.setTimeout(() => {
    persistTimer = null;
    if (pending == null) return;
    const text = pending;
    writeSceneFile(text)
      .then(() => {
        // Only clear dirty if nothing changed while the write was in flight.
        if (useStore.getState().sceneJson === text) {
          useStore.getState().setSceneDirty(false);
        }
      })
      .catch((e) => console.error('scene persist:', e));
  }, PERSIST_DEBOUNCE_MS);
}

/** Parse → clone → mutate → commit. No-op when scene.json doesn't parse. */
export function commitSceneMutation(mutator: (scene: any) => any) {
  const raw = useStore.getState().sceneJson;
  let scene: any;
  try {
    scene = JSON.parse(raw);
  } catch {
    console.warn('scene.json does not parse — mutation skipped');
    return;
  }
  const next = mutator(structuredClone(scene)) ?? scene;
  commitSceneText(JSON.stringify(next, null, 2));
}
