// Step 3 — Surface: `rectify_surface` on the lit photo (+ the ambient one if
// pinned). Auto-detects the projector's throw, or uses the quad you drag; the
// run writes three canvas-sized files into surfaces/ (`<run>_lit.png` = the
// lit crop, `<run>_surface.png` = the dark colour-corrected surface,
// `<run>_ambient.png`). Pin the best run ("Use this run") or single outputs
// (★). Every past run is a trial: click one to load its outputs, its settings
// and its corners back.

import { useEffect, useMemo, useState } from 'react';
import { libraryItems, libraryRecord } from '../../api/hub';
import { pinAsset, rectifySurface, type ProjectInfo, type RectifyResult } from '../../api/project';
import { pinsByPath, refreshProjectInfo } from '../../state/projectAssets';
import {
  SURFACE_DEFAULTS,
  patchSurface,
  surfaceDraft,
  useSetup,
  type Corners,
  type SurfaceSettings,
} from '../../state/setup';
import { useStore } from '../../state/store';
import { Btn } from '../../ui/primitives';
import { Lightbox } from '../../ui/media/Lightbox';
import { MediaGallery } from '../../ui/media/MediaGallery';
import { MediaTile } from '../../ui/media/MediaTile';
import type { MediaRef } from '../../ui/media/types';
import { Note, SectionTitle, Slider, Spinner, StepBody, baseName } from './common';
import { DEFAULT_QUAD, QuadEditor } from './QuadEditor';

type Output = { role: 'surface_lit' | 'surface' | 'surface_ambient'; label: string; what: string; path: string };

export function SurfaceStep({ info }: { info: ProjectInfo }) {
  const project = info.project;
  const draft = useSetup((s) => surfaceDraft(s, project));
  const { settings, corners, manual, run } = draft;
  const lit = info.assets.photo_lit;
  const ambient = info.assets.photo_ambient;
  const [useAmbient, setUseAmbient] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lb, setLb] = useState<number | null>(null);
  const [trialNote, setTrialNote] = useState<string | null>(null);
  const pins = useMemo(() => pinsByPath(info), [info]);

  // A different lit photo makes old corners meaningless.
  useEffect(() => {
    if (lit && draft.litPath !== lit.path) patchSurface(project, { litPath: lit.path, corners: null, manual: false });
  }, [lit?.path]);

  const set = (patch: Partial<SurfaceSettings>) => patchSurface(project, { settings: { ...settings, ...patch } });

  const go = async () => {
    if (!lit) return;
    setBusy(true);
    setErr(null);
    setTrialNote(null);
    try {
      const r = await rectifySurface({
        project,
        lit_photo: lit.path,
        ambient_photo: useAmbient && ambient ? ambient.path : null,
        corners: manual && corners ? corners : null,
        ...settings,
      });
      patchSurface(project, { run: r, corners: r.corners, manual: r.manual_corners });
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const outputs: Output[] = run
    ? ([
        { role: 'surface_lit', label: 'Lit crop', what: 'Shape input, segmentation input', path: run.surface_lit },
        { role: 'surface', label: 'Surface', what: 'Dark, colour-corrected — image_1, subtraction background', path: run.surface },
        run.surface_ambient
          ? { role: 'surface_ambient', label: 'Ambient crop', what: 'The room-lit view, aligned', path: run.surface_ambient }
          : null,
      ].filter(Boolean) as Output[])
    : [];
  const refs: MediaRef[] = outputs.map((o) => ({ path: o.path, name: o.label, kind: 'image' }));
  const runPinned = outputs.length > 0 && outputs.every((o) => info.assets[o.role]?.path === o.path);

  const pin = async (list: Output[]) => {
    try {
      for (const o of list) await pinAsset(project, o.role, o.path);
      refreshProjectInfo(project);
      useStore.getState().toast(list.length > 1 ? `Pinned run ${run?.run}` : `Pinned ${list[0].role}`, 'ok');
    } catch (e) {
      useStore.getState().toast(`Pin failed: ${e}`, 'bad');
    }
  };

  /** A trial tile (`<run>_surface.png`) → its outputs + settings + corners. */
  const loadTrial = async (ref: MediaRef) => {
    const m = ref.path.match(/^(.*)\/([^/]+)_surface\.png$/);
    if (!m) return;
    const [, dir, name] = m;
    const lit_ = `${dir}/${name}_lit.png`;
    const amb = `${dir}/${name}_ambient.png`;
    let hasAmb = false;
    try {
      hasAmb = (await libraryItems([amb]))[0]?.exists ?? false;
    } catch {
      /* keep false */
    }
    let args: Record<string, any> = {};
    let rec: Record<string, any> = {};
    const id = ref.item?.render_id;
    if (id) {
      try {
        const r = await libraryRecord(id);
        rec = r.record ?? {};
        args = rec.args ?? r.job_args ?? {};
      } catch (e) {
        console.warn('trial record', e);
      }
    }
    const next: SurfaceSettings = { ...SURFACE_DEFAULTS };
    for (const k of Object.keys(next) as (keyof SurfaceSettings)[]) {
      if (typeof args[k] === 'number') next[k] = args[k];
    }
    const recCorners: Corners | null =
      (Array.isArray(args.corners) && args.corners.length === 4 ? args.corners : null) ??
      (Array.isArray(rec.corners) ? rec.corners : null) ??
      (Array.isArray(rec.result?.corners) ? rec.result.corners : null);
    const loaded: RectifyResult = {
      run: name,
      surface_lit: lit_,
      surface: ref.path,
      surface_ambient: hasAmb ? amb : null,
      corners: recCorners ?? [],
      manual_corners: Array.isArray(args.corners),
      align: null,
      size: [ref.w ?? 0, ref.h ?? 0],
    };
    patchSurface(project, {
      run: loaded,
      settings: next,
      corners: recCorners,
      manual: Array.isArray(args.corners),
    });
    setTrialNote(
      args.lit_photo && lit && args.lit_photo !== lit.path
        ? `Run ${name} was made from another lit photo (${baseName(String(args.lit_photo))}).`
        : !id
        ? `Run ${name}: no render record found, so its settings weren't loaded.`
        : null
    );
  };

  if (!lit) {
    return (
      <StepBody
        main={<Note tone="warn">Pin a lit photo in the Photos step first — this step crops and darkens it.</Note>}
        side={<span className="text-[11px] text-fg3">Settings appear once there's a lit photo.</span>}
      />
    );
  }

  return (
    <StepBody
      main={
        <>
          <SectionTitle
            right={
              <div className="flex items-center gap-1.5">
                <span className={'text-[10px] font-semibold tracking-wider ' + (manual ? 'text-amber' : 'text-fg3')}>
                  {manual ? 'MANUAL CORNERS' : corners ? 'AUTO-DETECTED' : 'AUTO-DETECT'}
                </span>
                {manual ? (
                  <Btn
                    size="xs"
                    onClick={() =>
                      patchSurface(project, { manual: false, corners: run && !run.manual_corners && run.corners.length === 4 ? run.corners : null })
                    }
                    title="Forget the hand-placed corners; the next run detects the throw itself"
                  >
                    Auto-detect
                  </Btn>
                ) : !corners ? (
                  <Btn size="xs" onClick={() => patchSurface(project, { corners: DEFAULT_QUAD, manual: true })} title="Place the four corners yourself">
                    Set corners by hand
                  </Btn>
                ) : null}
              </div>
            }
          >
            Lit photo
          </SectionTitle>
          <QuadEditor
            path={lit.path}
            corners={corners}
            dim={busy}
            onChange={(c) => patchSurface(project, { corners: c, manual: true })}
          />
          <p className="text-[10px] text-fg3 -mt-2">
            Drag a corner to correct the detection (⇧ = fine, loupe follows). Click a corner, then arrow keys nudge it by a pixel.
          </p>

          {err && <Note tone="bad">{err}</Note>}
          {trialNote && <Note tone="warn">{trialNote}</Note>}

          {run && (
            <div className="flex flex-col gap-2">
              <SectionTitle
                right={
                  <Btn size="sm" tone="deck" active={!runPinned} disabled={runPinned} onClick={() => pin(outputs)} title="Pin every output of this run as the show's surface assets">
                    {runPinned ? '✓ This run is pinned' : 'Use this run'}
                  </Btn>
                }
              >
                Run {run.run}
                {run.align?.method && (
                  <span className="ml-2 text-[10px] font-normal text-fg3">
                    ambient aligned · {run.align.method}
                    {run.align.confidence != null ? ` · ${(run.align.confidence * 100).toFixed(0)} %` : ''}
                  </span>
                )}
              </SectionTitle>
              <div className="grid grid-cols-3 gap-3">
                {outputs.map((o, i) => (
                  <div key={o.role} className="flex flex-col gap-1 min-w-0">
                    <MediaTile
                      media={refs[i]}
                      onClick={() => setLb(i)}
                      onOpen={() => setLb(i)}
                      pins={pins[o.path]}
                      onPin={() => pin([o])}
                    />
                    <div className="flex items-baseline gap-1.5 min-w-0">
                      <span className="text-[11px] font-medium">{o.label}</span>
                      <span className="text-[10px] text-fg3 truncate">{o.what}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-col gap-2">
            <SectionTitle right={<span className="text-[10px] text-fg3">Click a run to load it back · ✕ deletes it</span>}>
              Trials
            </SectionTitle>
            <MediaGallery
              dir={info.layout.surfaces?.path}
              filter={(it) => it.name.endsWith('_surface.png')}
              project={project}
              roles={['surface']}
              selectedPath={run?.surface ?? null}
              onSelect={loadTrial}
              onPinned={() => refreshProjectInfo(project)}
              compareWith={lit.path}
              allowDelete
              minTile={140}
              emptyText="No runs yet — Run makes the first one."
            />
          </div>

          {lb != null && (
            <Lightbox
              items={refs}
              index={lb}
              onIndex={setLb}
              onClose={() => setLb(null)}
              compareWith={outputs[lb]?.role === 'surface_lit' ? null : run?.surface_lit}
              actions={(ref) => {
                const o = outputs.find((x) => x.path === ref.path);
                if (!o) return null;
                const pinned = info.assets[o.role]?.path === o.path;
                return (
                  <Btn size="sm" disabled={pinned} onClick={() => pin([o])}>
                    {pinned ? `✓ ${o.role}` : `★ Pin as ${o.role}`}
                  </Btn>
                );
              }}
            />
          )}
        </>
      }
      side={
        <>
          <span className="label">Detection</span>
          <Slider
            label="Margin"
            value={settings.margin}
            min={0}
            max={0.1}
            step={0.0025}
            def={SURFACE_DEFAULTS.margin}
            pct
            digits={3}
            onChange={(v) => set({ margin: v })}
            hint="Shrinks the detected quad by this much so the crop stays inside the light"
          />
          <Slider
            label="Min. projection size"
            value={settings.min_projection_fraction}
            min={0.05}
            max={0.6}
            step={0.01}
            def={SURFACE_DEFAULTS.min_projection_fraction}
            pct
            onChange={(v) => set({ min_projection_fraction: v })}
            hint="The throw must cover at least this share of the photo — raise it if detection grabs a small bright spot"
          />
          <span className="label pt-2 border-t border-line">Darken</span>
          <Slider
            label="Max brightness"
            value={settings.max_brightness}
            min={0.05}
            max={0.6}
            step={0.01}
            def={SURFACE_DEFAULTS.max_brightness}
            onChange={(v) => set({ max_brightness: v })}
            hint="Luminance ceiling of the dark surface (0–1)"
          />
          <Slider
            label="Detail boost"
            value={settings.detail_boost}
            min={0.5}
            max={2.5}
            step={0.05}
            def={SURFACE_DEFAULTS.detail_boost}
            onChange={(v) => set({ detail_boost: v })}
            hint="1 keeps the texture as it is, more sharpens it"
          />
          <Slider
            label="Chroma correction"
            value={settings.chroma_correction}
            min={0}
            max={1}
            step={0.05}
            def={SURFACE_DEFAULTS.chroma_correction}
            onChange={(v) => set({ chroma_correction: v })}
            hint="0 keeps the colours, 1 dims them with the brightness"
          />
          <Slider
            label="Gradient source"
            value={settings.gradient_source}
            min={0}
            max={1}
            step={0.05}
            def={SURFACE_DEFAULTS.gradient_source}
            onChange={(v) => set({ gradient_source: v })}
            hint="0 weights structural edges, 1 weights texture in dark areas"
          />
          <div className="flex items-center gap-2">
            <Btn size="xs" tone="ghost" onClick={() => set(SURFACE_DEFAULTS)}>
              Defaults
            </Btn>
            <span className="text-[10px] text-fg4">double-click a slider resets it</span>
          </div>
          <label className="flex items-start gap-2 pt-2 border-t border-line cursor-pointer">
            <input
              type="checkbox"
              checked={useAmbient && !!ambient}
              disabled={!ambient}
              onChange={(e) => setUseAmbient(e.target.checked)}
              className="mt-[2px] accent-[rgb(var(--deck))]"
            />
            <span className="text-[11px] leading-snug">
              <span className="block font-medium">Use the ambient photo</span>
              <span className="block text-fg3">
                {ambient ? 'Darken the room-lit view (aligned to the lit crop) instead of the lit crop.' : 'None pinned — the lit crop is darkened.'}
              </span>
            </span>
          </label>
          <Btn tone="deck" active size="lg" disabled={busy} onClick={go} className="w-full mt-1">
            {busy ? (
              <>
                <Spinner /> Running…
              </>
            ) : (
              `Run${manual ? ' with these corners' : ''}`
            )}
          </Btn>
        </>
      }
    />
  );
}
