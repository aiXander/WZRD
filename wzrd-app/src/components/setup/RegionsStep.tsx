// Step 5 — Regions: the colour map → one binary mask per region
// (`extract_color_regions` into regions/<run>/), named here, then
// `build_layerpack` into packs/ — which also writes a starter scene.json when
// the show has none, so the show can be opened in the engine.
//
// The last extract per show is kept in state/setup.ts, so jumping between
// steps doesn't lose ids and labels you typed.

import { useEffect, useRef, useState } from 'react';
import { mediaUrl } from '../../api/hub';
import { runTool, type ProjectInfo, type RegionsResult } from '../../api/project';
import { openProject } from '../../state/actions';
import { showName } from '../../state/hub';
import { refreshProjectInfo } from '../../state/projectAssets';
import {
  REGIONS_DEFAULTS,
  closeSetup,
  patchRegions,
  regionsDraft,
  slugId,
  useSetup,
  type RegionsSettings,
} from '../../state/setup';
import { useStore } from '../../state/store';
import { Btn } from '../../ui/primitives';
import { AssetPicker } from '../../ui/media/AssetPicker';
import { Field, Img, Note, SectionTitle, Slider, Spinner, StepBody } from './common';

const ID_RE = /^[a-z0-9][a-z0-9_-]{0,47}$/;
const FALLBACK = ['#ff4a52', '#3cc8ff', '#3ee08f', '#ffb02e', '#b48cff', '#ff7ad9', '#7af0ff', '#d4ff4a'];

export function RegionsStep({ info }: { info: ProjectInfo }) {
  const project = info.project;
  const d = useSetup((s) => regionsDraft(s, project));
  const [busy, setBusy] = useState<'extract' | 'build' | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [armOpen, setArmOpen] = useState(false);
  const set = (p: Parameters<typeof patchRegions>[1]) => patchRegions(project, p);
  const setS = (p: Partial<RegionsSettings>) => set({ settings: { ...d.settings, ...p } });

  const seg = d.segmentation ?? info.assets.segmentation?.path ?? null;
  const surface = d.surface ?? info.assets.surface?.path ?? null;
  const regions = d.result?.regions ?? [];

  const extract = async () => {
    if (!seg) return;
    setBusy('extract');
    setErr(null);
    try {
      const r = await runTool<RegionsResult>('extract_color_regions', {
        image: seg,
        ...(surface ? { surface_image: surface } : {}),
        ...d.settings,
        project,
      });
      // Keep typed ids/labels when the region count didn't change.
      const keep = d.ids.length === r.regions.length;
      set({
        result: r,
        ids: keep ? d.ids : r.regions.map((_, i) => `region_${i + 1}`),
        labels: keep ? d.labels : r.regions.map(() => ''),
        built: null,
      });
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  };

  const idProblems = d.ids.map((id, i) =>
    !ID_RE.test(id) ? 'lowercase letters, digits, _ or -' : d.ids.indexOf(id) !== i ? 'already used' : null
  );
  const idsOk = regions.length > 0 && idProblems.every((p) => !p);
  const packName = slugId(d.packName);

  const build = async () => {
    if (!d.result || !idsOk) return;
    setBusy('build');
    setErr(null);
    try {
      const layers: Record<string, { label: string }> = {};
      d.ids.forEach((id, i) => {
        const label = d.labels[i]?.trim();
        if (label) layers[id] = { label };
      });
      const r = await runTool<{ pack_dir: string; layer_count: number; note?: string; scene_json?: string }>('build_layerpack', {
        masks: regions.map((x) => x.region_mask),
        mask_ids: d.ids,
        ...(surface ? { surface_image: surface } : {}),
        ...(info.assets.photo_lit ? { source_capture: info.assets.photo_lit.path } : {}),
        ...(Object.keys(layers).length ? { tags: { layers } } : {}),
        ...(packName ? { pack_name: packName } : {}),
        project,
      });
      set({ built: { pack_dir: r.pack_dir, layer_count: r.layer_count, note: r.note, scene_json: r.scene_json } });
      await refreshProjectInfo(project);
      useStore.getState().toast(`Built a ${r.layer_count}-layer pack`, 'ok');
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  };

  const isOpen = showName() === project;
  const openShow = async () => {
    if (!armOpen) return setArmOpen(true);
    await openProject(info.scene_path);
    closeSetup();
  };

  return (
    <StepBody
      main={
        <>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Segmentation map">
              <AssetPicker project={project} value={seg} onChange={(p) => set({ segmentation: p })} suggest={['segmentation']} />
            </Field>
            <Field label="Surface" hint="Region crops are cut from it; it's the pack's preview photo.">
              <AssetPicker project={project} value={surface} onChange={(p) => set({ surface: p })} suggest={['surface', 'surface_lit']} />
            </Field>
          </div>
          {err && <Note tone="bad">{err}</Note>}
          {!d.result ? (
            <Note>
              {seg ? 'Extract splits the map into one mask per colour.' : 'Pin a segmentation map in the previous step (or pick one above).'}
            </Note>
          ) : (
            <>
              <SectionTitle right={<span className="text-[10px] text-fg3">{regions.length} regions · {d.result.regions_dir}</span>}>
                Overview
              </SectionTitle>
              <Overview surface={surface} regions={d.result} hover={hover} />
              <SectionTitle right={<span className="text-[10px] text-fg3">ids are what scenes and clips target — keep them stable</span>}>
                Name the regions
              </SectionTitle>
              <div className="grid gap-2.5" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
                {regions.map((r, i) => {
                  const color = r.color ?? FALLBACK[i % FALLBACK.length];
                  return (
                    <div
                      key={r.region_mask}
                      onMouseEnter={() => setHover(i)}
                      onMouseLeave={() => setHover((h) => (h === i ? null : h))}
                      className={'flex gap-2 p-2 rounded-md border bg-panel2/50 ' + (hover === i ? 'border-line2' : 'border-line')}
                    >
                      {/* The region's shape in its colour: colour fill × the (white-on-black) crop mask. */}
                      <div
                        className="relative w-[72px] h-[72px] shrink-0 rounded overflow-hidden border border-line isolate"
                        style={{ background: color }}
                        title={r.region_surface_image ?? r.region_mask}
                      >
                        <Img path={r.region_crop_mask ?? r.region_mask} className="w-full h-full object-contain bg-black mix-blend-multiply" />
                      </div>
                      <div className="flex flex-col gap-1 min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="w-3 h-3 rounded-sm border border-line2 shrink-0" style={{ background: color }} />
                          <span className="num text-[10px] text-fg3">
                            {r.area_fraction != null ? `${(r.area_fraction * 100).toFixed(1)} %` : ''}
                          </span>
                        </div>
                        <input
                          value={d.ids[i] ?? ''}
                          spellCheck={false}
                          onChange={(e) => {
                            const ids = [...d.ids];
                            ids[i] = e.target.value.toLowerCase().replace(/\s+/g, '_');
                            set({ ids });
                          }}
                          title={idProblems[i] ?? 'Region id'}
                          className={
                            'h-6 px-1.5 rounded bg-bg border text-[11px] font-mono outline-none ' +
                            (idProblems[i] ? 'border-bad' : 'border-line focus:border-line2')
                          }
                        />
                        <input
                          value={d.labels[i] ?? ''}
                          placeholder="label (optional)"
                          onChange={(e) => {
                            const labels = [...d.labels];
                            labels[i] = e.target.value;
                            set({ labels });
                          }}
                          className="h-6 px-1.5 rounded bg-bg border border-line text-[11px] outline-none focus:border-line2"
                        />
                        {idProblems[i] && <span className="text-[9px] text-bad">{idProblems[i]}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="flex items-end gap-3 pt-3 border-t border-line">
                <Field label="Pack name" hint={packName ? `packs/${packName}/` : 'Empty = a timestamped folder'}>
                  <input
                    value={d.packName}
                    onChange={(e) => set({ packName: e.target.value })}
                    placeholder="e.g. main"
                    className="h-7 w-[200px] px-2 rounded-md bg-bg border border-line text-[12px] outline-none"
                  />
                </Field>
                <Btn tone="deck" active disabled={!idsOk || !!busy} onClick={build} className="mb-[18px]">
                  {busy === 'build' ? (
                    <>
                      <Spinner /> Building…
                    </>
                  ) : (
                    `Build layer pack (${regions.length})`
                  )}
                </Btn>
              </div>
              {d.built && (
                <Note tone={d.built.note ? 'warn' : 'ok'}>
                  <div>
                    Built <b>{d.built.layer_count}</b> layers into <span className="text-fg2">{d.built.pack_dir}</span>.
                  </div>
                  {d.built.note && <div className="mt-1">{d.built.note}</div>}
                  {info.has_scene && (
                    <div className="flex items-center gap-2 mt-2">
                      <Btn size="sm" tone={armOpen ? 'amber' : 'deck'} active onClick={openShow}>
                        {armOpen ? 'Click again — the wall goes dark a moment' : isOpen ? 'Reopen show' : 'Open show'}
                      </Btn>
                      <span className="text-[10px] text-fg3">Loads {project} in the engine (Prepare / Perform).</span>
                    </div>
                  )}
                </Note>
              )}
            </>
          )}
        </>
      }
      side={
        <>
          <span className="label">Extraction</span>
          <Slider
            label="Max colours"
            value={d.settings.max_colors}
            min={2}
            max={16}
            step={1}
            digits={0}
            def={REGIONS_DEFAULTS.max_colors}
            onChange={(v) => setS({ max_colors: v })}
            hint="Upper cap on distinct colours kept"
          />
          <Slider
            label="Min. region size"
            value={d.settings.min_area_fraction}
            min={0.002}
            max={0.1}
            step={0.001}
            pct
            digits={3}
            def={REGIONS_DEFAULTS.min_area_fraction}
            onChange={(v) => setS({ min_area_fraction: v })}
            hint="Regions smaller than this share of the image are dropped"
          />
          <Slider
            label="Colour merge (ΔE)"
            value={d.settings.delta_e_threshold}
            min={1}
            max={20}
            step={0.5}
            digits={1}
            def={REGIONS_DEFAULTS.delta_e_threshold}
            onChange={(v) => setS({ delta_e_threshold: v })}
            hint="Colours closer than this merge — lower splits subtler differences"
          />
          <Slider
            label="Background (ΔE)"
            value={d.settings.background_threshold}
            min={5}
            max={40}
            step={1}
            digits={0}
            def={REGIONS_DEFAULTS.background_threshold}
            onChange={(v) => setS({ background_threshold: v })}
            hint="Pixels this close to the background (black) count as background"
          />
          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={d.settings.merge_same_color}
              onChange={(e) => setS({ merge_same_color: e.target.checked })}
              className="mt-[2px] accent-[rgb(var(--deck))]"
            />
            <span className="text-[11px] leading-snug">
              <span className="block font-medium">Merge same colour</span>
              <span className="block text-fg3">Disconnected blobs of one colour become one region.</span>
            </span>
          </label>
          <Btn tone="deck" active size="lg" className="w-full mt-1" disabled={!seg || !!busy} onClick={extract}>
            {busy === 'extract' ? (
              <>
                <Spinner /> Extracting…
              </>
            ) : d.result ? (
              'Extract again'
            ) : (
              'Extract regions'
            )}
          </Btn>
        </>
      }
    />
  );
}

/** The surface with every region mask added on top in its colour (the
 *  hovered one brighter). Pure canvas compositing — media URLs are another
 *  origin, so pixels are never read back (a tainted canvas can still draw). */
function Overview({ surface, regions, hover }: { surface: string | null; regions: RegionsResult; hover: number | null }) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const [layers, setLayers] = useState<{ base: HTMLImageElement | null; tints: HTMLCanvasElement[]; w: number; h: number } | null>(null);

  useEffect(() => {
    let dead = false;
    const load = (p: string) =>
      new Promise<HTMLImageElement>((res, rej) => {
        const img = new Image();
        img.onload = () => res(img);
        img.onerror = () => rej(new Error(`can't load ${p}`));
        img.src = mediaUrl(p);
      });
    (async () => {
      try {
        const masks = await Promise.all(regions.regions.map((r) => load(r.region_mask)));
        const base = surface ? await load(surface).catch(() => null) : null;
        const ref = base ?? masks[0];
        if (!ref) return;
        const scale = Math.min(1, 1100 / ref.naturalWidth);
        const w = Math.round(ref.naturalWidth * scale);
        const h = Math.round(ref.naturalHeight * scale);
        const tints = masks.map((m, i) => {
          const c = document.createElement('canvas');
          c.width = w;
          c.height = h;
          const x = c.getContext('2d')!;
          x.fillStyle = regions.regions[i].color ?? FALLBACK[i % FALLBACK.length];
          x.fillRect(0, 0, w, h);
          x.globalCompositeOperation = 'multiply';
          x.drawImage(m, 0, 0, w, h);
          return c;
        });
        if (!dead) setLayers({ base, tints, w, h });
      } catch (e) {
        console.warn('regions overview', e);
      }
    })();
    return () => {
      dead = true;
    };
  }, [regions, surface]);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !layers) return;
    c.width = layers.w;
    c.height = layers.h;
    const x = c.getContext('2d')!;
    x.globalCompositeOperation = 'source-over';
    x.globalAlpha = 1;
    x.fillStyle = 'black';
    x.fillRect(0, 0, layers.w, layers.h);
    if (layers.base) {
      // The dark surface is dim by design (max ≈ 0.2): drawn as is.
      x.globalAlpha = 1;
      x.drawImage(layers.base, 0, 0, layers.w, layers.h);
    }
    x.globalCompositeOperation = 'lighter';
    layers.tints.forEach((t, i) => {
      x.globalAlpha = hover == null ? 0.6 : hover === i ? 0.95 : 0.18;
      x.drawImage(t, 0, 0);
    });
  }, [layers, hover]);

  return (
    // shrink-0: inside the step's flex column the box would otherwise be
    // squeezed below the canvas's aspect and the image cropped/squashed.
    <div className="shrink-0 rounded-md border border-line bg-black overflow-hidden flex items-center justify-center">
      <canvas
        ref={canvas}
        className="block max-w-full max-h-[380px]"
        style={layers ? { aspectRatio: `${layers.w} / ${layers.h}` } : undefined}
      />
    </div>
  );
}
