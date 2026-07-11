// Debounced scene-commit path shared by every widget that mutates
// scene.json (binding inspector, driver rack, surface canvas).
//
// Semantics:
//   - The store's `sceneJson` updates immediately (optimistic — the UI never
//     waits on the engine).
//   - The engine push (`scene.load`, full plan rebuild) trails by ~150 ms so
//     a slider drag or rapid keystrokes collapse into one rebuild.
//   - The disk write trails the *accepted* push by ~650 ms and ONLY happens
//     for scenes the engine accepted. A rejected draft stays in memory (the
//     widgets keep showing it, the Reload pill shows FAIL) but never reaches
//     disk — the file must stay last-good so a restart always recovers.
//     (2026-07-11: persisting rejected scenes let one bad edit brick every
//     subsequent boot into a white projector window.)
//   - The engine-side reload dedupe skips the file-watcher echo of our own
//     write.

import { sceneLoad, writeSceneFile } from '../api/ipc';
import { useStore } from './store';

let pushTimer: number | null = null;
let persistTimer: number | null = null;
let pending: string | null = null;

const PUSH_DEBOUNCE_MS = 150;
const PERSIST_AFTER_ACCEPT_MS = 650;

export function commitSceneText(text: string) {
  const st = useStore.getState();
  st.setSceneJson(text);
  st.setSceneDirty(true);
  pending = text;

  if (pushTimer != null) window.clearTimeout(pushTimer);
  pushTimer = window.setTimeout(() => {
    pushTimer = null;
    if (pending == null) return;
    const text = pending;
    sceneLoad(text)
      .then(() => schedulePersist(text))
      .catch((e) => {
        // Rejected: reflect it on the Reload pill immediately (the engine
        // also emits a sticky hot_reload FAIL, this just beats the fan-out)
        // and leave the disk file untouched.
        console.error('scene apply:', e);
        useStore.getState().setHotReload({
          target: 'scene',
          ok: false,
          elapsed_ms: 0,
          message: String(e),
        });
      });
  }, PUSH_DEBOUNCE_MS);
}

/** Persist an engine-accepted scene text to disk, debounced. Skipped when a
 *  newer edit superseded it — that edit's own push cycle persists (or holds
 *  back) the newer text. */
function schedulePersist(text: string) {
  if (persistTimer != null) window.clearTimeout(persistTimer);
  persistTimer = window.setTimeout(() => {
    persistTimer = null;
    if (pending !== text) return;
    writeSceneFile(text)
      .then(() => {
        // Only clear dirty if nothing changed while the write was in flight.
        if (useStore.getState().sceneJson === text) {
          useStore.getState().setSceneDirty(false);
        }
      })
      .catch((e) => console.error('scene persist:', e));
  }, PERSIST_AFTER_ACCEPT_MS);
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
