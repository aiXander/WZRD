// Prepare — daylight-me's workbench (spec §7: "preparation, not
// configuration"). Always the DESIGN leg: nothing here touches the wall
// until TAKE.
//
//   ┌ layers ┬ surface map (primary) ─────────────┬ inspector ┐
//   │ regions│                                     │ target    │
//   │        │                                     │ effect    │
//   │        │                                     │ dials     │
//   └────────┴─ drawer: clip grid | code (scene.json + WGSL) ┴┘
//
// Building a layer is two clicks: a region on the map, an effect card.
// Putting a rendered clip on a region is one: its slot in the clip grid.

import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { nativeUnderlay } from '../api/previewMode';
import { SurfaceMap } from '../components/SurfaceMap';
import { ClipGrid } from '../components/ClipGrid';
import { LayerList, RegionList } from '../components/LayerList';
import { DarkRegionInspector, LayerInspector, defaultsFor } from '../components/LayerInspector';
import { clearSelection, setDeck } from '../state/actions';
import { commitSceneMutation } from '../state/sceneCommit';
import { regionName, selectorCovers, uniqueId, useScene } from '../state/scene';
import { useStore } from '../state/store';
import { Btn, Panel } from '../ui/primitives';
import { Pane, Split } from '../ui/layoutSplit';

const MonacoPanel = lazy(() => import('../components/MonacoPanel').then((m) => ({ default: m.MonacoPanel })));

export function Prepare() {
  const pack = useStore((s) => s.pack);
  const catalog = useStore((s) => s.catalog);
  const selected = useStore((s) => s.selectedBindingId);
  const selectedRegion = useStore((s) => s.selectedLayerId);
  const setSelected = useStore((s) => s.setSelectedBindingId);
  const setRegion = useStore((s) => s.setSelectedLayerId);
  const setHovered = useStore((s) => s.setHoveredBindingId);
  const toast = useStore((s) => s.toast);
  const scene = useScene();
  const bindings = scene?.bindings ?? [];
  const [armed, setArmed] = useState(false);
  const [drawer, setDrawerState] = useState<'clips' | 'code' | null>(() => {
    const v = localStorage.getItem('wzrd.prepare.drawer');
    return v === 'code' || v === 'none' ? (v === 'none' ? null : 'code') : 'clips';
  });
  const setDrawer = (d: 'clips' | 'code' | null) => {
    const next = drawer === d ? null : d;
    localStorage.setItem('wzrd.prepare.drawer', next ?? 'none');
    setDrawerState(next);
  };

  // Authoring is a DESIGN act — put the hands there on the way in.
  useEffect(() => {
    if (useStore.getState().deck?.preview_source === 'live') {
      setDeck('design');
      toast('Hands on DESIGN for authoring — the wall is untouched until TAKE', 'info');
    }
  }, [toast]);
  useEffect(() => () => setHovered(null), [setHovered]);
  // Esc also disarms "+ New" (the selection half lives in shortcuts.ts).
  useEffect(() => {
    if (!armed) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setArmed(false);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [armed]);

  const defaultEffect = useMemo(() => {
    const users = Object.entries(catalog).filter(([, e]) => e.kind === 'user').map(([n]) => n);
    return users.sort()[0] ?? 'tint';
  }, [catalog]);

  function createLayer(regionId: string) {
    const label = regionName(pack, regionId);
    const id = uniqueId(label, new Set(bindings.map((b) => b.id)));
    commitSceneMutation((s) => {
      s.bindings = [
        ...(s.bindings ?? []),
        {
          id,
          select: { id: regionId },
          effect: defaultEffect,
          params: defaultsFor(catalog[defaultEffect]?.inputs ?? []),
        },
      ];
      return s;
    });
    setSelected(id);
    setRegion(regionId);
    setArmed(false);
  }

  // Map + region-list clicks. Empty map / the margin around it (null)
  // clears both selections. A new region selects it and the first layer
  // lighting it (or keeps the current layer if that one covers it too);
  // re-clicking the selected region steps through its layers and, past the
  // last, deselects — the pack's full-frame "background" region means
  // almost every click hits *some* region, so "click again" is the way out.
  function onRegionClick(regionId: string | null) {
    if (!regionId || !pack) return clearSelection();
    if (armed) return createLayer(regionId);
    const layer = pack.layers.find((l) => l.id === regionId)!;
    const covering = bindings.filter((b) => selectorCovers(b.select, pack, layer));
    const i = covering.findIndex((b) => b.id === selected);
    if (regionId !== selectedRegion) {
      setRegion(regionId);
      setSelected(i >= 0 ? selected : covering[0]?.id ?? null);
    } else if (i + 1 < covering.length) {
      setSelected(covering[i + 1].id);
    } else {
      clearSelection();
    }
  }

  const current = bindings.find((b) => b.id === selected) ?? null;
  const darkRegion =
    !current && selectedRegion && pack && !bindings.some((b) => selectorCovers(b.select, pack, pack.layers.find((l) => l.id === selectedRegion)!))
      ? selectedRegion
      : null;

  return (
    // Outer skeleton = resizable splits (ui/layoutSplit.tsx; unlock in the ⚙
    // menu, saved per project). Default sizes = the old fixed grid.
    <Split id="prepare.rows" dir="col" rearrange={false} className="h-full p-2">
      <Pane id="cols" fill min={240} label="Workbench">
      <Split id="prepare.cols" dir="row">
        {/* Left: layers + regions */}
        <Pane id="left" size="248px" min={180} label="Layers + regions">
        <Split id="prepare.left" dir="col">
          <Pane id="layers" fill min={100} label="Layers">
          <LayerList
            bindings={bindings}
            armed={armed}
            onNew={() => {
              if (selectedRegion && !armed) createLayer(selectedRegion);
              else setArmed((a) => !a);
            }}
            className="flex-[3]"
          />
          </Pane>
          <Pane id="regions" size="40%" min={80} label="Regions">
          <Panel title="Regions" className="flex-[2]" bodyClassName="overflow-auto">
            <RegionList bindings={bindings} onPick={onRegionClick} onCreate={createLayer} />
          </Panel>
          </Pane>
        </Split>
        </Pane>

        {/* Center: the surface */}
        <Pane id="surface" fill min={320} label="Surface">
        <Panel hole={nativeUnderlay()} className="overflow-hidden" bodyClassName="min-h-0 flex flex-col">
          <div className="flex-1 min-h-0">
            <SurfaceMap bindings={bindings} onRegionClick={onRegionClick} armed={armed} />
          </div>
          <div className="flex items-center gap-2 h-8 px-2 border-t border-line shrink-0">
            <Btn size="sm" tone="ghost" active={drawer === 'clips'} onClick={() => setDrawer('clips')} title="Rendered clips per region — click one to play it on DESIGN">
              ▦ Clips
            </Btn>
            <Btn size="sm" tone="ghost" active={drawer === 'code'} onClick={() => setDrawer('code')} title="scene.json + effect shaders">
              {'</>'} Code
            </Btn>
            <span className="text-[10px] text-fg4 truncate min-w-0">
              {drawer === 'clips'
                ? 'Rendered videos, a row per region. Click = play it there in DESIGN (one per row) · click again or ■ = stop · TAKE sends it to the wall.'
                : 'Plain files: scene.json + effects/*/shader.wgsl. ⌘S (or Save) in the editor saves the open file; a shader save hot-reloads.'}
            </span>
          </div>
        </Panel>
        </Pane>

        {/* Right: inspector */}
        <Pane id="inspector" size="340px" min={260} label="Inspector">
        <Panel title={current ? 'Layer' : 'Inspector'} bodyClassName="overflow-auto">
          {current ? (
            <LayerInspector binding={current} bindings={bindings} />
          ) : darkRegion ? (
            <DarkRegionInspector regionId={darkRegion} onCreate={createLayer} />
          ) : (
            <div className="p-4 text-[12px] text-fg3 leading-relaxed">
              Pick a layer on the left, or click any region on the surface. Click a selected one again (or Esc) to
              deselect.
            </div>
          )}
        </Panel>
        </Pane>
      </Split>
      </Pane>

      {drawer === 'code' && (
        <Pane id="code" size="42%" min={160} label="Code">
          <div className="rounded-lg border border-line overflow-hidden">
            <Suspense fallback={<div className="p-3 text-[11px] text-fg3">loading editor…</div>}>
              <MonacoPanel />
            </Suspense>
          </div>
        </Pane>
      )}
      {drawer === 'clips' && (
        <Pane id="clips" size="34%" min={120} label="Clips">
          <div className="rounded-lg border border-line bg-panel overflow-hidden">
            <ClipGrid />
          </div>
        </Pane>
      )}
    </Split>
  );
}
