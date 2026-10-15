// The hub's health lamp (studio plan A1) — sits with the rig's other lamps
// in the TopBar. Green = attached; amber = starting; red = down (click:
// start it) or port 8787 held by something that isn't a WZRD hub. The
// popover says why and offers Restart hub.

import { useRef, useState } from 'react';
import { ensureHub, restartHub, useHub } from '../state/hub';
import { Btn, Dot } from '../ui/primitives';
import { Popover } from '../ui/Popover';

const WORD: Record<string, string> = {
  up: 'Hub',
  starting: 'Hub…',
  down: 'Hub down',
  foreign: 'Port taken',
};

export function HubLamp() {
  const status = useHub((s) => s.status);
  const anchor = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);
  const state = status?.state ?? 'starting';
  const tone = state === 'up' ? 'ok' : state === 'starting' ? 'warn' : 'bad';
  const busy = state === 'starting';
  return (
    <>
      <button
        ref={anchor}
        onClick={() => setOpen((o) => !o)}
        title={
          state === 'up'
            ? `Content hub running (pid ${status?.pid ?? '?'}) — renders, library, labels`
            : `Content hub ${state}${status?.error ? ': ' + status.error : ''}`
        }
        className={
          'flex items-center gap-1.5 px-2 xl:px-2.5 h-8 shrink-0 rounded-lg bg-bg border border-line text-[11px] ' +
          (state === 'down' || state === 'foreign' ? 'text-bad font-semibold' : 'text-fg2 hover:text-fg')
        }
      >
        <Dot tone={tone} />
        <span className={state === 'up' ? 'hidden xl:inline' : ''}>{WORD[state]}</span>
      </button>
      {open && (
        <Popover anchor={anchor.current} onClose={() => setOpen(false)} width={320}>
          <div className="label px-1 pb-1.5">Content hub</div>
          <div className="px-1 text-[12px] leading-snug">
            {state === 'up' && (
              <>
                Running · pid <span className="num">{status?.pid}</span>
                {status?.version ? <> · v{status.version}</> : null}
              </>
            )}
            {state === 'starting' && 'Starting…'}
            {state === 'down' && 'Not running. Studio still browses the library; renders and labels need the hub.'}
            {state === 'foreign' && 'Something else holds port 8787 — quit it, then start the hub.'}
          </div>
          {status?.error && state !== 'up' && (
            <div className="mt-1.5 px-1 text-[10px] text-fg3 break-words">{status.error}</div>
          )}
          {status?.media_root && (
            <div className="mt-2 px-1 text-[10px] text-fg3 truncate" title={status.media_root}>
              Library: {status.media_root}
            </div>
          )}
          <div className="flex items-center gap-1.5 px-1 pt-3">
            {state === 'up' ? (
              <Btn
                size="sm"
                disabled={busy}
                onClick={() => restartHub()}
                title="Stop the hub and start a fresh one. Running renders are lost."
              >
                Restart hub
              </Btn>
            ) : (
              <Btn size="sm" disabled={busy} onClick={() => ensureHub()} title="Attach to a running hub, or start one">
                {busy ? 'Starting…' : 'Start hub'}
              </Btn>
            )}
          </div>
        </Popover>
      )}
    </>
  );
}
