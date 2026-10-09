// Transport seam under `ipc.ts`. Inside the Tauri shell every command is a
// Tauri `invoke` and every event a Tauri `listen` — the production path.
//
// Outside Tauri (the same Vite build opened in a plain browser) the UI talks
// straight to the engine's JSON-RPC WebSocket (`ws://127.0.0.1:9123`, the
// same §2.3 surface the shell proxies) and to a dev-only Vite middleware for
// the handful of commands that are file I/O in the shell. That lets the UI
// be iterated and driven from a browser against a live engine; it is a dev
// harness and a remote-operator sketch, not a shipped mode.

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export const IN_TAURI =
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

export function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (IN_TAURI) return invoke<T>(cmd, args);
  return import('./wsBridge').then((b) => b.bridgeCall<T>(cmd, args ?? {}));
}

export function listenEvent<T>(
  event: string,
  handler: (payload: T) => void
): Promise<UnlistenFn> {
  if (IN_TAURI) return listen<T>(event, (evt) => handler(evt.payload));
  return import('./wsBridge').then((b) => b.bridgeListen<T>(event, handler));
}
