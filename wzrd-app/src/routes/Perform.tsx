// Perform — the booth. Laid out like a VJ desk:
//
//   ┌ monitor (what your hands are on) ─────────┬ deck: DESIGN|LIVE, TAKE ┐
//   ├ looks strip (fire whole scenes) ──────────┤ masters + audio         │
//   ├ Layers | Clips (one tabbed panel) ┬ dials  │                         │
//
// Every edge is a resizable split and every panel can swap places while the
// layout is unlocked (⚙ menu; `ui/layoutSplit.tsx`, saved per project).
// The monitor frame wears the deck colour, so a glance tells you whether
// you're touching the wall or the draft.

import { useState } from 'react';
import { NativePreview } from '../components/NativePreview';
import { DeckPanel } from '../components/DeckPanel';
import { AudioPanel, MastersPanel } from '../components/MastersPanel';
import { LayerStack } from '../components/LayerStack';
import { LayerParams } from '../components/LayerParams';
import { LooksBank } from '../components/LooksBank';
import { ClipGrid } from '../components/ClipGrid';
import { useStore } from '../state/store';
import { Panel, Seg } from '../ui/primitives';
import { Pane, Split } from '../ui/layoutSplit';

export function Perform() {
  const leg = useStore((s) => s.deck?.preview_source ?? 'design');
  const pack = useStore((s) => s.pack);
  return (
    <Split id="perform.cols" dir="row" className="h-full p-2">
      <Pane id="main" fill min={480} label="Stage">
        <Split id="perform.main" dir="col">
          <Pane id="monitor" fill min={140} label="Monitor">
            <section
              className={
                'relative min-h-0 rounded-lg border-2 overflow-hidden flex flex-col ' +
                (leg === 'live' ? 'border-live/80' : 'border-design/60')
              }
            >
              <div className="flex items-center gap-2 h-7 px-2.5 shrink-0 bg-panel border-b border-line">
                <span
                  className={
                    'text-[10px] font-bold tracking-[0.16em] px-1.5 rounded-sm ' +
                    (leg === 'live' ? 'bg-live text-white' : 'bg-design text-black')
                  }
                >
                  {leg === 'live' ? 'PROGRAM' : 'PREVIEW'}
                </span>
                <span className="text-[11px] text-fg2 truncate">
                  {leg === 'live' ? 'What the wall shows — your hands are live' : 'Design draft — the crowd can’t see this'}
                </span>
                <div className="flex-1" />
                {pack && <span className="num text-[10px] text-fg3">{pack.width}×{pack.height}</span>}
                <FpsLabel />
              </div>
              <div className="flex-1 min-h-0 p-2">
                <NativePreview />
              </div>
            </section>
          </Pane>

          <Pane id="looks" size="62px" min={44} label="Looks">
            <div className="flex rounded-lg border border-line bg-panel overflow-hidden">
              <LooksBank />
            </div>
          </Pane>

          <Pane id="bottom" size="38%" min={180} label="Mixer + dials">
            <Split id="perform.bottom" dir="row">
              <Pane id="mixer" fill min={260} label="Layers / Clips">
                <MixerPanel />
              </Pane>
              <Pane id="dials" size="46%" min={240} label="Dials">
                <Panel bodyClassName="min-h-0">
                  <LayerParams />
                </Panel>
              </Pane>
            </Split>
          </Pane>
        </Split>
      </Pane>

      <Pane id="side" size="316px" min={260} label="Deck + masters">
        <div className="min-h-0 flex flex-col gap-2">
          <Panel title="Deck" className={leg === 'live' ? 'border-live/50' : 'border-design/40'}>
            <DeckPanel />
          </Panel>
          <Panel
            title="Masters"
            right={<span className="text-[10px] text-fg4">both legs · the room</span>}
            className="flex-1 min-h-[300px]"
            bodyClassName="min-h-0 flex flex-col"
          >
            <div className="flex-1 min-h-0">
              <MastersPanel />
            </div>
            <div className="border-t border-line">
              <AudioPanel />
            </div>
          </Panel>
        </div>
      </Pane>
    </Split>
  );
}

type MixerTab = 'layers' | 'clips';
const TAB_KEY = 'wzrd.perform.mixerTab';

/** Layers (the mixer) and Clips share one full-height panel behind a tab —
 *  stacked, each showed ~2 rows. The tab is remembered per machine. Only the
 *  visible one is mounted (the clip grid's 2 s poll runs only while shown). */
function MixerPanel() {
  const leg = useStore((s) => s.deck?.preview_source ?? 'design');
  const [tab, setTabState] = useState<MixerTab>(() => {
    try {
      return localStorage.getItem(TAB_KEY) === 'clips' ? 'clips' : 'layers';
    } catch {
      return 'layers';
    }
  });
  const setTab = (t: MixerTab) => {
    try {
      localStorage.setItem(TAB_KEY, t);
    } catch {
      /* private mode — fine */
    }
    setTabState(t);
  };
  return (
    <Panel
      title={
        <>
          <Seg<MixerTab>
            size="xs"
            value={tab}
            onChange={setTab}
            options={[
              { id: 'layers', label: 'Layers', title: 'The layer mixer: opacity, mute, solo' },
              { id: 'clips', label: 'Clips', title: 'Rendered clips per region — click one to play it on DESIGN' },
            ]}
          />
          <span className={'normal-case tracking-normal font-medium ' + (leg === 'live' ? 'text-live' : 'text-design')}>
            · {tab === 'clips' ? 'design' : leg === 'live' ? 'live' : 'design'}
          </span>
        </>
      }
      right={
        <span className="text-[10px] text-fg4 truncate">
          {tab === 'layers' ? 'M mute · S solo · [ ] fade' : 'click = on DESIGN · T to take'}
        </span>
      }
      bodyClassName={tab === 'layers' ? 'overflow-auto' : 'min-h-0'}
    >
      {tab === 'layers' ? <LayerStack /> : <ClipGrid compact />}
    </Panel>
  );
}

/** The monitor header's fps readout — its own component so the 2 Hz `fps`
 *  channel re-renders this label, not the whole room. */
function FpsLabel() {
  const fps = useStore((s) => (s.fps ? Math.round(s.fps.fps) : null));
  if (fps == null) return null;
  return <span className="num text-[10px] text-fg3">{fps} fps</span>;
}
