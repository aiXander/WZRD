// Perform — the booth. Laid out like a VJ desk:
//
//   ┌ monitor (what your hands are on) ─────────┬ deck: DESIGN|LIVE, TAKE ┐
//   ├ looks bank (fire whole scenes) ───────────┤ masters + audio         │
//   ├ layer stack (mixer) ─┬ selected layer's dials                       │
//
// The monitor frame wears the deck colour, so a glance tells you whether
// you're touching the wall or the draft.

import { NativePreview } from '../components/NativePreview';
import { DeckPanel } from '../components/DeckPanel';
import { AudioPanel, MastersPanel } from '../components/MastersPanel';
import { LayerStack } from '../components/LayerStack';
import { LayerParams } from '../components/LayerParams';
import { LooksBank } from '../components/LooksBank';
import { useStore } from '../state/store';
import { Panel } from '../ui/primitives';

export function Perform() {
  const leg = useStore((s) => s.deck?.preview_source ?? 'design');
  const pack = useStore((s) => s.pack);
  const fps = useStore((s) => s.fps);
  return (
    <div className="h-full min-h-0 grid grid-cols-[minmax(0,1fr)_316px] gap-2 p-2">
      <div className="min-h-0 min-w-0 grid grid-rows-[minmax(0,1fr)_112px_minmax(250px,36%)] gap-2">
        {/* Monitor */}
        <section
          className={
            'relative min-h-0 rounded-lg border-2 bg-black overflow-hidden flex flex-col ' +
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
            <span className="text-[11px] text-fg2">
              {leg === 'live' ? 'What the wall shows — your hands are live' : 'Design draft — the crowd can’t see this'}
            </span>
            <div className="flex-1" />
            {pack && <span className="num text-[10px] text-fg3">{pack.width}×{pack.height}</span>}
            {fps && <span className="num text-[10px] text-fg3">{Math.round(fps.fps)} fps</span>}
          </div>
          <div className="flex-1 min-h-0 p-2">
            <NativePreview />
          </div>
        </section>

        <Panel className="overflow-hidden" bodyClassName="min-h-0">
          <LooksBank />
        </Panel>

        <div className="min-h-0 grid grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] gap-2">
          <Panel
            title={
              <>
                Layers
                <span className={'normal-case tracking-normal font-medium ' + (leg === 'live' ? 'text-live' : 'text-design')}>
                  · {leg === 'live' ? 'live' : 'design'}
                </span>
              </>
            }
            right={<span className="text-[10px] text-fg4">M mute · S solo · [ ] fade</span>}
            bodyClassName="overflow-auto"
          >
            <LayerStack />
          </Panel>
          <Panel bodyClassName="min-h-0">
            <LayerParams />
          </Panel>
        </div>
      </div>

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
    </div>
  );
}
