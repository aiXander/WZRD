// Finder drops into the app (show-setup M1). Tauri's webview drag-drop event
// gives absolute paths (an HTML5 drop doesn't). One Tauri listener for the
// whole page; users form a **stack**: only the most recently mounted
// *enabled* user gets hover + drops (a dialog's drop zone wins over the room
// behind it). No-op in the browser seat (no Tauri, no paths).

import { useEffect, useRef, useState } from 'react';
import { IN_TAURI } from '../../api/transport';

type User = { id: number; enabled: boolean; onPaths: (p: string[]) => void; setHover: (h: boolean) => void };

const users: User[] = [];
let seq = 0;
let unlisten: (() => void) | null = null;
let installing = false;

const top = () => [...users].reverse().find((u) => u.enabled) ?? null;

function hoverOnly(u: User | null) {
  for (const x of users) x.setHover(x === u);
}

async function install() {
  if (unlisten || installing || !IN_TAURI) return;
  installing = true;
  try {
    const { getCurrentWebview } = await import('@tauri-apps/api/webview');
    const un = await getCurrentWebview().onDragDropEvent((ev) => {
      const p = ev.payload;
      if (p.type === 'enter' || p.type === 'over') hoverOnly(top());
      else if (p.type === 'leave') hoverOnly(null);
      else if (p.type === 'drop') {
        hoverOnly(null);
        const u = top();
        if (u && p.paths.length) u.onPaths(p.paths);
      }
    });
    if (users.length) unlisten = un;
    else un();
  } catch (e) {
    console.warn('file drop listener:', e);
  } finally {
    installing = false;
  }
}

export function useFileDrop(onPaths: (paths: string[]) => void, enabled = true): { hovering: boolean } {
  const [hovering, setHovering] = useState(false);
  const user = useRef<User | null>(null);
  if (!user.current) user.current = { id: ++seq, enabled, onPaths, setHover: setHovering };
  user.current.onPaths = onPaths;
  user.current.enabled = enabled;

  useEffect(() => {
    if (!IN_TAURI) return;
    const u = user.current!;
    users.push(u);
    install();
    return () => {
      const i = users.indexOf(u);
      if (i >= 0) users.splice(i, 1);
      if (!users.length && unlisten) {
        unlisten();
        unlisten = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!enabled) setHovering(false);
  }, [enabled]);

  return { hovering: enabled && hovering };
}
