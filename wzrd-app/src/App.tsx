// Application root.
//   - Wires the engine (telemetry → store; status → store).
//   - Hydrates pack, both legs' scenes, override tables, effect catalog and
//     looks on first mount, and from scratch whenever the engine comes back
//     (crash restart, project switch — A1 re-roots the engine in-process).
//   - Renders the top bar + the active room, the keyboard map, toasts.
//
// Nothing modal mid-set: errors surface inline or as a toast.

import { useEffect, useRef } from 'react';
import { useStore, type Catalog, type MastersState } from './state/store';
import {
  effectDescribe,
  engineStatus,
  lastPayload,
  listEffects,
  masterSet,
  transportSet,
  onEngineStatus,
  onProject,
  onTelemetry,
  packInfo,
  projectCurrent,
  readSceneFile,
  sceneGetState,
  type AlignmentDoc,
  type ChangeEntry,
  type DeckPayload,
  type PerfSpike,
} from './api/ipc';
import { ingestAlignment } from './state/alignment';
import { hydrateOverrides, loadLooks, refreshLiveScene, resetActionState } from './state/actions';
import { dropPendingCommits } from './state/sceneCommit';
import { useAutopilot } from './state/autopilot';
import { useDecksLoader, usePlayedSignals } from './state/decks';
import { useHubWire } from './state/hub';
import { ensureMasks } from './state/masks';
import { loadLayout } from './ui/layoutStore';
import { useShortcuts } from './state/shortcuts';
import { useAutopilotDriver } from './state/autopilot';
import { HelpOverlay, Toasts, TopBar } from './components/TopBar';
import { EngineDownBanner } from './components/ProjectMenu';
import { SetupHost } from './components/setup/SetupWizard';
import { Studio } from './routes/Studio';
import { Prepare } from './routes/Prepare';
import { Align } from './routes/Align';
import { Perform } from './routes/Perform';
import { DebugRoute } from './routes/Debug';

async function loadCatalog() {
  try {
    const res: any = await effectDescribe();
    const next: Catalog = {};
    for (const e of res?.effects ?? []) next[e.name] = { kind: e.kind, inputs: e.inputs ?? [] };
    useStore.getState().setCatalog(next);
  } catch (e) {
    console.warn('effect.describe', e);
  }
}

/** Run `hydrate` now, or once more after the in-flight one — a re-root
 *  fires both the status edge and `engine:project`. */
let hydrating: Promise<void> | null = null;
let hydrateAgain = false;
function rehydrate() {
  if (hydrating) {
    hydrateAgain = true;
    return;
  }
  hydrating = hydrate().finally(() => {
    hydrating = null;
    if (hydrateAgain) {
      hydrateAgain = false;
      rehydrate();
    }
  });
}

async function hydrate() {
  // Everything below belongs to one engine instance: start from scratch so
  // nothing of a previous project (or a crashed engine) leaks in.
  dropPendingCommits();
  resetActionState();
  const prev = useStore.getState().project;
  useStore.getState().resetProject();
  const st = useStore.getState();
  try {
    const p = await projectCurrent();
    st.setProject(p);
    // Panel layout is per project (ui_layout.json); no-op for the same one.
    loadLayout(p.scene_path);
    if (prev && prev.scene_path !== p.scene_path) useAutopilot.getState().setOn(false);
  } catch (e) {
    console.warn('project_current:', e);
  }
  try {
    const s = await engineStatus();
    st.setEngineRunning(s.running);
    st.setEngineError(s.running ? null : s.last_error);
    if (!s.running) return;
  } catch (e) {
    console.warn('engine_status:', e);
  }
  try {
    st.setPack(await packInfo());
  } catch (e) {
    console.warn('pack_info:', e);
  }
  // Design truth is the engine's design leg (it may hold a restored draft
  // that differs from scene.json); fall back to the file.
  try {
    const d = await sceneGetState('design');
    st.setSceneJson(d.json);
  } catch {
    try {
      st.setSceneJson(await readSceneFile());
    } catch (e) {
      console.warn('read_scene_file:', e);
    }
  }
  refreshLiveScene();
  try {
    st.setEffects(await listEffects());
  } catch (e) {
    console.warn('list_effects:', e);
  }
  loadCatalog();
  loadLooks();
  // Sticky channels emitted before the webview mounted.
  try {
    const m = await lastPayload<MastersState>('masters');
    if (m) st.setMasters(m);
  } catch (e) {
    console.warn('last_payload(masters):', e);
  }
  try {
    const d = await lastPayload<DeckPayload>('deck');
    if (d) st.setDeck(d);
  } catch (e) {
    console.warn('last_payload(deck):', e);
  }
  try {
    const s = await lastPayload<PerfSpike>('perf_spike');
    if (s) st.setPerfSpike(s);
  } catch (e) {
    console.warn('last_payload(perf_spike):', e);
  }
  try {
    const a = await lastPayload<AlignmentDoc>('alignment');
    if (a) st.setAlignment(a);
  } catch (e) {
    console.warn('last_payload(alignment):', e);
  }
  hydrateOverrides('design');
  hydrateOverrides('live');
  lockLegs();
}

/** Masters are the room, not a leg (actions.setMaster writes both). Older
 *  sessions may have left the legs apart: copy live's masters onto design
 *  and snap design's clock to live's, so they tick in lockstep from here
 *  and a TAKE never jumps the wall's speed, brightness or bar grid. */
function lockLegs() {
  const st = useStore.getState();
  const m = st.masters;
  if (!st.deck?.two_leg || !m) return;
  (Object.keys(m.live) as (keyof typeof m.live)[]).forEach((k) => {
    if (m.design[k] !== m.live[k]) masterSet(k, m.live[k], 'design').catch(() => {});
  });
  st.setMasters({ ...m, design: { ...m.live } });
  transportSet({ sync: true }).catch((e) => console.warn('transport sync', e));
}

export default function App() {
  const route = useStore((s) => s.route);
  const leg = useStore((s) => s.deck?.preview_source ?? 'design');
  const pack = useStore((s) => s.pack);
  const lastPromote = useRef<string>('idle');
  useShortcuts();
  useAutopilotDriver();
  useDecksLoader();
  usePlayedSignals();
  useHubWire();

  // The whole room re-tints with the deck (see index.css `--deck`).
  useEffect(() => {
    document.documentElement.dataset.deck = leg;
  }, [leg]);

  useEffect(() => {
    ensureMasks(pack);
  }, [pack]);

  useEffect(() => {
    let unlistenT: (() => void) | undefined;
    let unlistenS: (() => void) | undefined;
    let unlistenP: (() => void) | undefined;
    (async () => {
      rehydrate();
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
          case 'transport':
            st.setTransport(frame.payload);
            break;
          case 'frame_stats':
            st.setFrameStats(frame.payload);
            break;
          case 'perf':
            st.setPerf(frame.payload);
            break;
          case 'perf_spike':
            st.setPerfSpike(frame.payload);
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
          case 'deck': {
            const d = frame.payload as DeckPayload;
            st.setDeck(d);
            // A take just landed: live now holds design's scene + controls.
            if (lastPromote.current !== 'idle' && d.promote === 'idle') {
              refreshLiveScene();
              hydrateOverrides('live');
            }
            lastPromote.current = d.promote;
            break;
          }
          case 'alignment':
            // The engine is the authority; `ingestAlignment` keeps facets
            // with an unacknowledged local drag.
            ingestAlignment(frame.payload);
            break;
          case 'preview':
            st.setPreview(frame.payload);
            break;
          case 'log':
            st.appendLog(frame.payload);
            break;
          case 'changes': {
            // §5.10 reverse-sync: re-pull the facet any non-UI writer touched.
            // Store only — persisting an agent-authored scene is the
            // operator's explicit "Adopt".
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
                loadCatalog();
              }
            }
            if (entry.actor === 'agent') st.setAgentEdit(entry);
            break;
          }
          default:
            break;
        }
      });
      unlistenS = await onEngineStatus((s: { running: boolean; last_error?: string | null }) => {
        const st = useStore.getState();
        const was = st.engineRunning;
        st.setEngineRunning(!!s.running);
        if (!s.running && s.last_error) st.setEngineError(s.last_error);
        if (s.running && !was) rehydrate();
      });
      // Tauri seat: every open/restart, including a failed one that kept
      // (or re-opened) the previous project.
      unlistenP = await onProject(() => rehydrate());
    })();
    return () => {
      unlistenT?.();
      unlistenS?.();
      unlistenP?.();
    };
  }, []);

  return (
    <div className="flex flex-col h-full">
      <TopBar />
      <EngineDownBanner />
      <main className="flex-1 min-h-0 overflow-hidden">
        {route === 'studio' && <Studio />}
        {route === 'prepare' && <Prepare />}
        {route === 'align' && <Align />}
        {route === 'perform' && <Perform />}
        {route === 'debug' && <DebugRoute />}
      </main>
      <HelpOverlay />
      <SetupHost />
      <Toasts />
    </div>
  );
}
