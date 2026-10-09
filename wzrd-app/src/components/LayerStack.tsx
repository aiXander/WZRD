// The layer stack — one row per binding, named in surface-language: the
// region's silhouette in its own colour, "Main facade · flow noise", and a
// mixer strip (mute, solo, opacity fader) that moves the leg your hands are
// on. Selecting a row aims the parameter panel at it (spec: "selecting a
// layer aims everything else").
//
// Rows come from the control leg's scene: on LIVE you see (and ride) what's
// on the wall; on DESIGN you see the draft, with NEW / EDITED tags against
// what's live.

import { useEffect, useMemo } from 'react';
import type { PackInfo } from '../api/ipc';
import { controlLeg, MIX, setOpacity, toggleMute, toggleSolo } from '../state/actions';
import { getMask, useMaskVersion } from '../state/masks';
import {
  bindingHue,
  canonical,
  effectName,
  isDriver,
  layerTitle,
  parseScene,
  prettify,
  regionsOf,
  targetLabel,
  type Binding,
} from '../state/scene';
import { useStore } from '../state/store';
import { Fader } from '../ui/Fader';

export function RegionGlyph({
  binding,
  pack,
  hue,
  size = 34,
  dim,
}: {
  binding: Binding;
  pack: PackInfo | null;
  hue: number;
  size?: number;
  dim?: boolean;
}) {
  useMaskVersion();
  const regs = regionsOf(binding, pack).slice(0, 6);
  const aspect = pack ? pack.width / pack.height : 4 / 3;
  return (
    <div
      className="relative shrink-0 rounded-[4px] bg-black border border-line overflow-hidden"
      style={{ width: size * aspect, height: size }}
    >
      {regs.map((r) => {
        const m = getMask(r.id);
        if (!m) return null;
        return (
          <div
            key={r.id}
            className="absolute inset-0"
            style={{
              background: `hsl(${hue} 85% 62%)`,
              opacity: dim ? 0.3 : 0.95,
              WebkitMaskImage: `url(${m.thumb})`,
              maskImage: `url(${m.thumb})`,
              WebkitMaskSize: '100% 100%',
              maskSize: '100% 100%',
            }}
          />
        );
      })}
    </div>
  );
}

function useLegScene() {
  const leg = useStore((s) => s.deck?.preview_source ?? 'design');
  const design = useStore((s) => s.sceneJson);
  const live = useStore((s) => s.liveSceneJson);
  return useMemo(
    () => ({ leg, scene: parseScene(leg === 'live' ? live || design : design), live: parseScene(live) }),
    [leg, design, live]
  );
}

export function LayerStack() {
  const pack = useStore((s) => s.pack);
  const { leg, scene, live } = useLegScene();
  const selected = useStore((s) => s.selectedBindingId);
  const setSelected = useStore((s) => s.setSelectedBindingId);
  const table = useStore((s) => s.overrides[leg]);
  const bindings = scene?.bindings ?? [];

  // Keep a selection whenever there's anything to select.
  useEffect(() => {
    if (bindings.length && !bindings.some((b) => b.id === selected)) setSelected(bindings[0].id);
  }, [bindings, selected, setSelected]);

  const liveById = useMemo(() => new Map((live?.bindings ?? []).map((b) => [b.id, canonical(JSON.stringify(b))])), [live]);
  const anySolo = bindings.some((b) => (table[b.id]?.[MIX.solo] ?? 0) > 0.5);

  if (!scene) return <div className="p-3 text-[11px] text-fg3">scene.json doesn’t parse — fix it in Prepare › Code.</div>;
  if (!bindings.length)
    return <div className="p-4 text-[12px] text-fg3">No layers yet. Build one in <b className="text-fg2">Prepare</b> (⌘1): pick a region, pick an effect.</div>;

  return (
    <div className="flex flex-col p-1.5 gap-1">
      {bindings.map((b, i) => {
        const row = table[b.id] ?? {};
        const opacity = row[MIX.opacity] ?? 1;
        const mute = (row[MIX.mute] ?? 0) > 0.5;
        const solo = (row[MIX.solo] ?? 0) > 0.5;
        const silenced = mute || (anySolo && !solo);
        const hue = bindingHue(b, pack, i);
        const sel = b.id === selected;
        const modded = Object.values(b.params ?? {}).some(isDriver);
        let tag: string | null = null;
        if (leg === 'design' && live) {
          const l = liveById.get(b.id);
          tag = !l ? 'NEW' : l !== canonical(JSON.stringify(b)) ? 'EDITED' : null;
        }
        return (
          <div
            key={b.id}
            onClick={() => setSelected(b.id)}
            className={
              'group grid grid-cols-[18px_auto_minmax(0,1fr)_auto_minmax(90px,38%)] items-center gap-2 h-11 pl-1.5 pr-2 rounded-md border transition-colors ' +
              (sel ? 'bg-raised border-deck/60' : 'bg-panel2/60 border-transparent hover:bg-panel2 hover:border-line')
            }
          >
            <div className="num text-[10px] text-fg4 text-center">{i + 1}</div>
            <RegionGlyph binding={b} pack={pack} hue={hue} dim={silenced} />
            <div className={'min-w-0 ' + (silenced ? 'opacity-40' : '')}>
              <div className="flex items-center gap-1.5">
                <span className="text-[12px] font-medium text-fg truncate">{layerTitle(b)}</span>
                {modded && <span className="text-[9px] font-semibold text-mod" title="audio / clock modulated">∿</span>}
                {tag && (
                  <span
                    className={
                      'text-[8px] font-bold tracking-wider px-1 rounded-sm ' +
                      (tag === 'NEW' ? 'bg-design/20 text-design' : 'bg-amber/20 text-amber')
                    }
                    title={tag === 'NEW' ? 'Only in design — TAKE to send it live' : 'Differs from the live version'}
                  >
                    {tag}
                  </span>
                )}
              </div>
              <div className="text-[10px] text-fg3 truncate">
                {targetLabel(b, pack)} · {prettify(effectName(b.effect))}
              </div>
            </div>
            <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
              <button
                title="Mute (M)"
                onClick={() => toggleMute(b.id)}
                className={
                  'w-6 h-6 rounded text-[10px] font-bold border ' +
                  (mute ? 'bg-bad/80 border-bad text-white' : 'border-line2 text-fg3 hover:text-fg hover:border-fg4')
                }
              >
                M
              </button>
              <button
                title="Solo (S) — ⇧-click to add to the solo set"
                onClick={(e) => toggleSolo(b.id, !e.shiftKey)}
                className={
                  'w-6 h-6 rounded text-[10px] font-bold border ' +
                  (solo ? 'bg-amber border-amber text-black' : 'border-line2 text-fg3 hover:text-fg hover:border-fg4')
                }
              >
                S
              </button>
            </div>
            <div onClick={(e) => e.stopPropagation()}>
              <Fader
                value={opacity}
                min={0}
                max={1}
                defaultValue={1}
                onChange={(v) => {
                  if (controlLeg() !== leg) return;
                  setOpacity(b.id, v);
                }}
                format={(v) => `${Math.round(v * 100)}`}
                color={`hsl(${hue} 85% 62%)`}
                size="sm"
                title="Layer opacity — double-click: 100%"
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
