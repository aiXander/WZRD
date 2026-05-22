// Top status strip — three pills the operator glances at during a show.
//
// Phase 4.1: OSC freshness, FPS, last-reload outcome.
// Phase 4.2: stays in the top bar across every route — the most important
// indicator during a live performance shouldn't disappear behind tabs.
//
// Click OSC pill → open audio-server browser UI (default http://127.0.0.1:8765
// per the audio server's `audio-server` CLI). Wired through the Tauri shell
// plugin so the link respects the user's default browser.

import { open as openExternal } from '@tauri-apps/plugin-shell';
import { useStore } from '../state/store';

function dotColor(state: 'fresh' | 'stale' | 'down' | null | undefined): string {
  switch (state) {
    case 'fresh':
      return 'bg-accent-green';
    case 'stale':
      return 'bg-accent-amber';
    case 'down':
      return 'bg-accent-red';
    default:
      return 'bg-ink-400';
  }
}

export function StatusStrip() {
  const fresh = useStore((s) => s.audioFresh);
  const fps = useStore((s) => s.fps);
  const hotReload = useStore((s) => s.hotReload);
  const engineRunning = useStore((s) => s.engineRunning);

  return (
    <div className="flex items-center gap-3 px-4 py-1.5 bg-ink-900 border-b border-ink-700">
      <button
        className="pill"
        title="Click to open the audio-server browser UI"
        onClick={() => {
          openExternal('http://127.0.0.1:8765/').catch(() => {});
        }}
      >
        <span className={`dot ${dotColor(fresh?.state)}`} />
        <span>OSC</span>
        <span className="text-zinc-400">
          {fresh?.state ?? 'unknown'}
          {fresh?.last_packet_ms ? ` · ${(fresh.last_packet_ms / 1000).toFixed(1)}s` : ''}
        </span>
      </button>

      <span className="pill">
        <span
          className={
            'dot ' +
            (engineRunning ? 'bg-accent-green' : 'bg-accent-red')
          }
        />
        <span>Engine</span>
        <span className="text-zinc-400">{engineRunning ? 'running' : 'down'}</span>
      </span>

      <span className="pill">
        <span className={`dot ${fps && fps.fps > 55 ? 'bg-accent-green' : 'bg-accent-amber'}`} />
        <span>FPS</span>
        <span className="text-zinc-400">
          {fps ? `${fps.fps.toFixed(0)} · ${fps.frame_time_ms.toFixed(1)}ms` : '—'}
        </span>
      </span>

      <span
        className="pill"
        title={hotReload?.message ?? undefined}
      >
        <span
          className={
            'dot ' +
            (hotReload == null
              ? 'bg-ink-400'
              : hotReload.ok
              ? 'bg-accent-green'
              : 'bg-accent-red')
          }
        />
        <span>Reload</span>
        <span className="text-zinc-400 truncate max-w-[24rem]">
          {hotReload
            ? `${hotReload.target} ${hotReload.ok ? 'OK' : 'FAIL'} ${
                hotReload.elapsed_ms > 0
                  ? `${hotReload.elapsed_ms.toFixed(0)}ms`
                  : ''
              }${hotReload.message ? ' — ' + hotReload.message.split('\n')[0] : ''}`
            : '—'}
        </span>
      </span>
    </div>
  );
}
