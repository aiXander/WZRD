// Application root.
//   - Sets up the engine wire (telemetry → store; status → store).
//   - Hydrates the pack + scene + effect list on first mount.
//   - Renders the top status strip + the active route (Phase 4.2 navigation).
//
// Single-window UI per the Phase 4 design principle: nothing modal, errors
// surface inline. Routes share the top status strip so it's always glanceable.

import { useEffect } from 'react';
import { useStore, type MastersState } from './state/store';
import {
  engineStatus,
  lastPayload,
  listEffects,
  onEngineStatus,
  onTelemetry,
  packInfo,
  readSceneFile,
  sceneGetState,
  type AlignmentDoc,
  type ChangeEntry,
  type DeckPayload,
} from './api/ipc';
import { ingestAlignment } from './state/alignment';
import { StatusStrip } from './components/StatusStrip';
import { TopBar } from './components/TopBar';
import { Prepare } from './routes/Prepare';
import { Align } from './routes/Align';
import { Perform } from './routes/Perform';
import { DebugRoute } from './routes/Debug';

export default function App() {
  const route = useStore((s) => s.route);
  const setRoute = useStore((s) => s.setRoute);

  useEffect(() => {
    let unlistenT: (() => void) | undefined;
    let unlistenS: (() => void) | undefined;
    (async () => {
      const tStore = useStore.getState();
      // Initial hydration. If the engine is still booting these will throw;
      // the status event will fire it again shortly.
      try {
        const s = await engineStatus();
        tStore.setEngineRunning(s.running);
      } catch (e) {
        console.warn('engine_status:', e);
      }
      try {
        const pack = await packInfo();
        tStore.setPack(pack);
      } catch (e) {
        console.warn('pack_info:', e);
      }
      try {
        const scene = await readSceneFile();
        tStore.setSceneJson(scene);
      } catch (e) {
        console.warn('read_scene_file:', e);
      }
      try {
        tStore.setEffects(await listEffects());
      } catch (e) {
        console.warn('list_effects:', e);
      }
      try {
        // Masters are sticky — the engine emitted them before the webview
        // mounted, so hydrate from the shell's last-payload snapshot.
        const m = await lastPayload<MastersState>('masters');
        if (m) tStore.setMasters(m);
      } catch (e) {
        console.warn('last_payload(masters):', e);
      }
      try {
        // §5.6 deck state is sticky too (emitted at design-leg boot).
        const d = await lastPayload<DeckPayload>('deck');
        if (d) tStore.setDeck(d);
      } catch (e) {
        console.warn('last_payload(deck):', e);
      }
      try {
        // §5.14 alignment is sticky too (emitted at boot, before the webview
        // exists) — so the Align tab hydrates without an extra round trip.
        const a = await lastPayload<AlignmentDoc>('alignment');
        if (a) tStore.setAlignment(a);
      } catch (e) {
        console.warn('last_payload(alignment):', e);
      }

      unlistenT = await onTelemetry((frame) => {
        const st = useStore.getState();
        switch (frame.channel) {
          case 'fps':
            st.setFps(frame.payload);
            break;
          case 'audio_freshness':
            st.setAudioFresh(frame.payload);
            break;
          case 'hot_reload':
            st.setHotReload(frame.payload);
            st.appendHotReload(frame.payload);
            break;
          case 'audio':
            st.setAudio(frame.payload);
            break;
          case 'frame_stats':
            st.setFrameStats(frame.payload);
            break;
          case 'drivers':
            st.setDrivers(frame.payload?.drivers ?? []);
            break;
          case 'connectivity':
            st.setConnectivity(frame.payload);
            break;
          case 'masters':
            st.setMasters(frame.payload);
            break;
          case 'deck':
            st.setDeck(frame.payload);
            break;
          case 'alignment':
            // §5.14 — the engine is the authority (a headless camera script is
            // as legitimate a writer as this UI). `ingestAlignment` takes it
            // wholesale except for facets with an unacknowledged local drag,
            // where the echo is the older view.
            ingestAlignment(frame.payload);
            break;
          case 'preview':
            st.setPreview(frame.payload);
            break;
          case 'log':
            st.appendLog(frame.payload);
            break;
          case 'changes': {
            // §5.10 reverse-sync: the store assumes it's the only writer, so
            // any non-UI design mutation (agent over WS, watcher file edit)
            // re-pulls the affected facet from the engine. Store only —
            // never into sceneCommit's disk debounce; persisting an
            // agent-authored scene is the operator's explicit "Adopt".
            const entry = frame.payload as ChangeEntry;
            if (entry.actor === 'ui') break;
            if (entry.facet === 'layers') {
              packInfo()
                .then((p) => useStore.getState().setPack(p))
                .catch((e) => console.warn('changes re-sync pack_info:', e));
            } else {
              sceneGetState()
                .then((r) => useStore.getState().setSceneJson(r.json))
                .catch((e) => console.warn('changes re-sync scene:', e));
              if (entry.facet === 'effects') {
                listEffects()
                  .then((eff) => useStore.getState().setEffects(eff))
                  .catch((e) => console.warn('changes re-sync effects:', e));
              }
            }
            if (entry.actor === 'agent') st.setAgentEdit(entry);
            break;
          }
          default:
            // Unknown channels are silently dropped — agent-driven additions
            // shouldn't need a UI patch to coexist.
            break;
        }
      });
      unlistenS = await onEngineStatus((s) => {
        useStore.getState().setEngineRunning(!!s.running);
      });
    })();

    return () => {
      unlistenT?.();
      unlistenS?.();
    };
  }, []);

  // ⌘1..⌘4 (and Ctrl on non-mac) switch routes, in load-in order.
  useEffect(() => {
    const KEYS: Record<string, ReturnType<typeof useStore.getState>['route']> = {
      '1': 'prepare',
      '2': 'align',
      '3': 'perform',
      '4': 'debug',
    };
    function onKey(e: KeyboardEvent) {
      if (!(e.metaKey || e.ctrlKey)) return;
      const target = KEYS[e.key];
      if (!target) return;
      e.preventDefault();
      setRoute(target);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setRoute]);

  return (
    <div className="flex flex-col h-full">
      <TopBar />
      <StatusStrip />
      <main className="flex-1 min-h-0 overflow-hidden">
        {route === 'prepare' && <Prepare />}
        {route === 'align' && <Align />}
        {route === 'perform' && <Perform />}
        {route === 'debug' && <DebugRoute />}
      </main>
    </div>
  );
}
