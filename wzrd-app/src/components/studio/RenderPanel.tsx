// Studio's Render panel (studio plan R1, S16): start renders from the app,
// no chat and no terminal. Every render is a non-blocking hub job (H2 —
// `POST /api/render`), visible in the jobs pill and as a tile in the folder
// it lands in (the target folder on the tray's "Renders land in" card), exactly like an agent's.
//
// - TextureFlow: a recipe (its style images + settings are the defaults;
//   what's set here wins), or the style tray's images; the target region
//   sets the aspect (long side = the default output width, snapped to its
//   step; pre-2.0 specs: `width`) and is recorded as "made for"; length /
//   motion / seed; Draft or Final; the backend — Modal | Local (the desktop
//   GPU, $0), each with its state lamp from `/api/tf/backends`, the choice
//   remembered per machine (`state/renderQueue.ts`), sent as `args.backend`.
//   A render is accepted whatever the backend's state: it waits in that
//   backend's lane of the render queue (`RenderQueue.tsx`). With a spec that has `draft` (≥ 2.2.0) a
//   Draft is the Final's own first pass (same settings + `draft: true`),
//   finished later from its record (Upscale, RecordPanel); older specs: the
//   fewest steps with the upscales off. Optional shape input
//   (`control_input`, an AssetPicker over the open show's named assets —
//   `surface_lit` / `surface` first) whose strength (full spec range; > 0
//   sends `use_controlnet1` + `controlnet_strength1`, 0 = no ControlNet),
//   and a `diffusion_mask`; each slot has a × to clear it.
//   Sent only when set (show-setup M1). Picking a region auto-fills both
//   with that region's crops (`/api/project/region_inputs`: its mask + the
//   show's surface cut to its padded `source_box`, the box the engine fits
//   the clip into) — overridable; a slot still holding the auto crop
//   follows the region. Every range / default / step comes
//   from the endpoint's arg spec (`/api/textureflow/spec`); the form stays
//   a curated subset of it.
// - Image (Nano Banana): a prompt + the tray as references.
// - Video (Kling v2.5): a prompt + the focused still as the start frame,
//   5 or 10 s.
// The cost estimate comes from the hub before Go (`/api/estimate`); past
// today's cap the app asks, and only then sends `allow_over_cap` (S15).

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  estimateRender,
  getTextureflowSpec,
  listRecipes,
  mediaUrl,
  startRender,
  type Estimate,
  type Recipe,
  type RenderTool,
  type TfParam,
  type TfSpec,
} from '../../api/hub';
import { refreshJobs, useHub } from '../../state/hub';
import {
  BACKENDS,
  BACKEND_LABEL,
  backendLook,
  preferredBackend,
  setPreferredBackend,
  startLocal,
  useQueue,
  useRenderQueue,
} from '../../state/renderQueue';
import { regionInputs, type RegionInputs } from '../../api/project';
import { regionName } from '../../state/scene';
import { useStore } from '../../state/store';
import { addStyles, itemByPath, short, useStudio } from '../../state/studio';
import { usePathDrop } from '../../ui/media/pathDrag';
import { isImagePath } from '../../ui/media/types';
import { Popover } from '../../ui/Popover';
import { Btn, Dot, Seg } from '../../ui/primitives';
import type { TfBackend } from '../../api/hub';
import { Fader } from '../../ui/Fader';
import { AssetPicker } from '../../ui/media/AssetPicker';
import { ModalSlots } from './RenderQueue';
import { SpendLine, wouldExceedCap } from './SpendLine';

type Kind = 'flow' | 'image' | 'video';
const TOOL: Record<Kind, RenderTool> = {
  flow: 'texture_flow',
  image: 'nano_banana',
  video: 'kling_v25_image_to_video',
};

/** A numeric spec field (`default`, `minimum`, …) of one TextureFlow arg. */
const num = (p: TfParam | undefined, key: 'default' | 'minimum' | 'maximum' | 'step'): number =>
  Number(p?.[key]);

/** A region's box → the TextureFlow output size: its aspect, long side = the
 *  spec's default width, snapped to its step and range. Spec ≥ 2.0.0 sizes
 *  the output (`output_width` / `output_height`); older ones `width` / `height`. */
function sizeFor(
  bbox: [number, number, number, number] | null | undefined,
  spec: TfSpec | null,
): Record<string, number> | null {
  const out = !!spec?.parameters.output_width;
  const [kw, kh] = out ? ['output_width', 'output_height'] : ['width', 'height'];
  const pw = spec?.parameters[kw];
  if (!bbox || !pw) return null;
  const w = Math.max(1, bbox[2] - bbox[0]);
  const h = Math.max(1, bbox[3] - bbox[1]);
  const [lo, hi, step, long] = [num(pw, 'minimum'), num(pw, 'maximum'), num(pw, 'step') || 1, num(pw, 'default')];
  const snap = (v: number) => Math.min(hi, Math.max(lo, lo + Math.round((v - lo) / step) * step));
  const [sw, sh] = w >= h ? [snap(long), snap((long * h) / w)] : [snap((long * w) / h), snap(long)];
  return { [kw]: sw, [kh]: sh };
}

export function RenderPanel() {
  const hubUp = useHub((s) => s.status?.state === 'up');
  const pack = useStore((s) => s.pack);
  const styles = useStudio((s) => s.selection.styles);
  const focus = useStudio((s) => s.focus);
  const [kind, setKind] = useState<Kind>('flow');
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [recipe, setRecipe] = useState<string>('');
  const [region, setRegion] = useState<string>('');
  const [tfSpec, setTfSpec] = useState<TfSpec | null>(null);
  // null = the spec's default (TextureFlow), until the operator moves the knob.
  const [secondsSet, setSeconds] = useState<number | null>(null);
  const [motionSet, setMotion] = useState<number | null>(null);
  const [seed, setSeed] = useState('');
  const [final, setFinal] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [count, setCount] = useState(1);
  const [est, setEst] = useState<Estimate | null>(null);
  const [estErr, setEstErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [overCap, setOverCap] = useState(false);
  const [controlInput, setControlInput] = useState<string | null>(null);
  const [cnStrengthSet, setCnStrength] = useState<number | null>(null);
  const [diffusionMask, setDiffusionMask] = useState<string | null>(null);
  // The region's auto crops; a slot still holding one follows the region.
  const [auto, setAuto] = useState<RegionInputs | null>(null);
  const autoRef = useRef<RegionInputs | null>(null);
  const [backend, setBackendState] = useState<TfBackend>(preferredBackend);
  const setBackend = (b: TfBackend) => {
    setBackendState(b);
    setPreferredBackend(b);
  };
  useRenderQueue(kind === 'flow');
  const showDir = useStore((s) => s.project?.dir);
  const show = showDir ? showDir.replace(/\/+$/, '').split('/').pop() ?? null : null;

  useEffect(() => {
    if (!hubUp) return;
    listRecipes()
      .then((r) => setRecipes(r.recipes))
      .catch((e) => console.warn('recipes', e));
    getTextureflowSpec()
      .then(setTfSpec)
      .catch((e) => console.warn('textureflow spec', e));
  }, [hubUp]);

  useEffect(() => {
    const prev = autoRef.current;
    const follow = (cur: string | null, was: string | null | undefined, next: string | null) =>
      cur == null || cur === was ? next : cur;
    const apply = (next: RegionInputs | null) => {
      autoRef.current = next;
      setAuto(next);
      setDiffusionMask((cur) => follow(cur, prev?.mask, next?.mask ?? null));
      setControlInput((cur) => follow(cur, prev?.shape, next?.shape ?? null));
    };
    if (!region || !show || !hubUp || kind !== 'flow') {
      if (!region) apply(null);
      return;
    }
    let live = true;
    regionInputs(show, region)
      .then((r) => live && apply(r))
      .catch((e) => {
        console.warn('region inputs', e);
        if (live) apply(null);
      });
    return () => {
      live = false;
    };
  }, [region, show, hubUp, kind]);

  const P = tfSpec?.parameters;
  const seconds = secondsSet ?? (kind === 'video' ? 5 : num(P?.n_seconds, 'default'));
  const motion = motionSet ?? num(P?.motion_scale, 'default');
  const cnStrength = cnStrengthSet ?? num(P?.controlnet_strength1, 'default');

  const focused = itemByPath(focus);
  const startFrame = focused?.kind === 'image' ? focused.path : null;
  const regionLayer = pack?.layers.find((l) => l.id === region);

  const args = useMemo((): Record<string, unknown> | null => {
    const common: Record<string, unknown> = region ? { region } : {};
    if (kind === 'flow') {
      if (!P || (!recipe && styles.length === 0)) return null;
      const a: Record<string, unknown> = { ...common, n_seconds: seconds, motion_scale: motion };
      if (recipe) a.recipe = recipe;
      if (styles.length) a.images = styles.slice(0, 6);
      const size = sizeFor(regionLayer?.source_box ?? regionLayer?.bbox, tfSpec);
      if (size) Object.assign(a, size);
      if (seed.trim()) a.seed = Number(seed);
      // A real draft (spec ≥ 2.2.0) is the final's first pass: same settings + draft.
      if (!final) Object.assign(a, P.draft ? { draft: true } : { n_steps: num(P.n_steps, 'minimum'), use_upscale: false, upscale_esrgan: false });
      if (controlInput) {
        a.control_input = controlInput;
        // Strength 0 = no ControlNet: the shape only shows through the mask's black areas.
        if (cnStrength > 0) Object.assign(a, { use_controlnet1: true, controlnet_strength1: Math.round(cnStrength * 100) / 100 });
      }
      if (diffusionMask) a.diffusion_mask = diffusionMask;
      a.backend = backend;
      return a;
    }
    if (!prompt.trim()) return null;
    if (kind === 'image') {
      return { ...common, prompt: prompt.trim(), ...(styles.length ? { image_urls: styles.slice(0, 10) } : {}) };
    }
    if (!startFrame) return null;
    return { ...common, prompt: prompt.trim(), image_url: startFrame, duration: seconds >= 8 ? '10' : '5' };
  }, [kind, recipe, styles, region, regionLayer, seconds, motion, seed, final, prompt, startFrame, controlInput, cnStrength, diffusionMask, backend, P, tfSpec]);

  // The estimate follows the form (debounced).
  const argsKey = JSON.stringify([kind, args, count]);
  useEffect(() => {
    setEst(null);
    setEstErr(null);
    setOverCap(false);
    if (!hubUp || !args) return;
    const t = setTimeout(() => {
      estimateRender(TOOL[kind], args, count)
        .then(setEst)
        .catch((e) => setEstErr(String(e)));
      // (to_hap has no estimate — it's free; it never runs from here.)
    }, 350);
    return () => clearTimeout(t);
  }, [argsKey, hubUp]);

  const wouldExceed = wouldExceedCap(est);

  const go = async (allowOverCap: boolean) => {
    if (!args) return;
    setBusy(true);
    try {
      const r = await startRender(TOOL[kind], allowOverCap ? { ...args, allow_over_cap: true } : args, count);
      const where = kind === 'flow' ? ` on ${BACKEND_LABEL[backend]}` : '';
      useStore.getState().toast(`Started ${r.jobs.length} ${kind === 'flow' ? 'TextureFlow' : kind} job${r.jobs.length === 1 ? '' : 's'}${where} → ${short(r.jobs[0]?.dest ?? '')}`, 'ok');
      setOverCap(false);
      refreshJobs();
    } catch (e) {
      useStore.getState().toast(`Render refused: ${e}`, 'bad');
    } finally {
      setBusy(false);
    }
  };

  if (!hubUp) {
    return <div className="p-3 text-[12px] text-fg3">Rendering needs the content hub — start it from the Hub lamp.</div>;
  }

  const missing =
    kind === 'flow'
      ? args
        ? null
        : !P
        ? 'Loading the TextureFlow spec from the hub…'
        : 'Pick a recipe or drop style images in the tray'
      : !prompt.trim()
      ? 'Write a prompt'
      : kind === 'video' && !startFrame
      ? 'Select a still in the grid as the start frame'
      : null;

  return (
    <div className="h-full min-h-0 flex flex-col">
      <div className="flex-1 min-h-0 overflow-y-auto p-3 flex flex-col gap-3 text-[12px]">
        <Seg<Kind>
          value={kind}
          onChange={setKind}
          options={[
            { id: 'flow', label: 'TextureFlow' },
            { id: 'image', label: 'Image' },
            { id: 'video', label: 'Video' },
          ]}
        />
        {kind === 'flow' && <BackendPick value={backend} onChange={setBackend} />}
        {kind === 'flow' && (
          <Field label="Recipe">
            <select value={recipe} onChange={(e) => setRecipe(e.target.value)} className="w-full h-7 bg-bg border border-line rounded px-1 text-[12px]">
              <option value="">— none (style tray only) —</option>
              {recipes.map((r) => (
                <option key={r.name} value={r.name} title={r.when_to_use}>
                  {r.name}
                  {r.look ? ` · ${r.look}` : ''}
                </option>
              ))}
            </select>
            {recipe && (
              <RecipeHint recipe={recipes.find((r) => r.name === recipe)} />
            )}
          </Field>
        )}
        {kind !== 'flow' && (
          <Field label="Prompt">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
              placeholder={kind === 'image' ? 'image_1 … refer to the tray images in order' : 'static camera, dark background, slow motion of …'}
              className="w-full bg-bg border border-line rounded px-2 py-1 text-[12px] outline-none resize-none"
            />
          </Field>
        )}
        <Field label={kind === 'video' ? 'Start frame' : 'Styles'}>
          {kind === 'video' ? (
            <span className="text-fg2 truncate block">{startFrame ? short(startFrame) : '— select a still in the grid —'}</span>
          ) : (
            <DropSlot onPaths={addStyles} accept={images}>
              <span className="text-fg2">
                {styles.length ? `${styles.length} from the tray${styles.length > (kind === 'flow' ? 6 : 10) ? ' (first ones used)' : ''}` : 'none in the tray — drag images here'}
                {kind === 'flow' && recipe && styles.length ? ' — override the recipe’s' : ''}
              </span>
            </DropSlot>
          )}
        </Field>
        <Field label="Made for">
          <select value={region} onChange={(e) => setRegion(e.target.value)} className="w-full h-7 bg-bg border border-line rounded px-1 text-[12px]">
            <option value="">— no region —</option>
            {(pack?.layers ?? []).map((l) => (
              <option key={l.id} value={l.id}>
                {regionName(pack, l.id)}
              </option>
            ))}
          </select>
          {kind === 'flow' && regionLayer && (
            <span className="text-[10px] text-fg3">
              {(() => {
                const s = sizeFor(regionLayer.source_box ?? regionLayer.bbox, tfSpec);
                return s ? `${Object.values(s).join('×')} (the region's aspect)` : '';
              })()}
            </span>
          )}
        </Field>
        {kind !== 'image' && (
          <Field label={`Length · ${Number.isFinite(seconds) ? seconds : '…'} s`}>
            <input
              type="range"
              min={kind === 'video' ? 5 : num(P?.n_seconds, 'minimum')}
              max={kind === 'video' ? 10 : num(P?.n_seconds, 'maximum')}
              step={kind === 'video' ? 5 : num(P?.n_seconds, 'step')}
              value={seconds}
              onChange={(e) => setSeconds(Number(e.target.value))}
              className="w-full accent-[rgb(var(--deck))]"
            />
          </Field>
        )}
        {kind === 'flow' && (
          <>
            <Field label={`Motion · ${Number.isFinite(motion) ? motion.toFixed(2) : '…'}`}>
              <input
                type="range"
                min={num(P?.motion_scale, 'minimum')}
                max={num(P?.motion_scale, 'maximum')}
                step={num(P?.motion_scale, 'step')}
                value={motion}
                onChange={(e) => setMotion(Number(e.target.value))}
                className="w-full accent-[rgb(var(--deck))]"
              />
            </Field>
            {show && (
              <>
                <Group label="Shape input">
                  <Clearable value={controlInput} onClear={() => setControlInput(null)}>
                    <DropSlot onPaths={(p) => setControlInput(p[0])} accept={images}>
                      <AssetPicker
                        project={show}
                        value={controlInput}
                        onChange={setControlInput}
                        suggest={['surface_lit', 'surface']}
                        placeholder="— none —"
                        compact
                      />
                    </DropSlot>
                  </Clearable>
                  <AutoHint auto={auto?.shape} value={controlInput} onUse={setControlInput} region={regionLayer && regionName(pack, regionLayer.id)} />
                  {controlInput && (
                    <Fader
                      value={cnStrength}
                      min={num(P?.controlnet_strength1, 'minimum')}
                      max={num(P?.controlnet_strength1, 'maximum')}
                      step={num(P?.controlnet_strength1, 'step')}
                      defaultValue={num(P?.controlnet_strength1, 'default')}
                      onChange={setCnStrength}
                      label="shape strength"
                      format={(v) => (v > 0 ? v.toFixed(2) : 'off')}
                      size="sm"
                      className="mt-0.5"
                      title={
                        'ControlNet: how closely the render follows this image\'s shapes / edges. 0 = off (the image then only ' +
                        'shows through the mask\'s black areas).' + (P?.controlnet_strength1?.description ? '\n\n' + P.controlnet_strength1.description : '')
                      }
                    />
                  )}
                </Group>
                <Group label="Mask (white = animated)">
                  <Clearable value={diffusionMask} onClear={() => setDiffusionMask(null)}>
                    <DropSlot onPaths={(p) => setDiffusionMask(p[0])} accept={images}>
                      <AssetPicker
                        project={show}
                        value={diffusionMask}
                        onChange={setDiffusionMask}
                        placeholder="— none —"
                        compact
                      />
                    </DropSlot>
                  </Clearable>
                  <AutoHint auto={auto?.mask} value={diffusionMask} onUse={setDiffusionMask} region={regionLayer && regionName(pack, regionLayer.id)} />
                </Group>
              </>
            )}
            <div className="flex gap-3">
              <Field label="Seed">
                <input
                  value={seed}
                  onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
                  placeholder="random"
                  className="w-24 h-7 bg-bg border border-line rounded px-2 text-[12px] num"
                />
              </Field>
              <Field label="Quality">
                <Seg<'draft' | 'final'>
                  value={final ? 'final' : 'draft'}
                  onChange={(v) => setFinal(v === 'final')}
                  options={
                    P?.draft
                      ? [
                          { id: 'draft', label: 'Draft · ½' },
                          { id: 'final', label: 'Final' },
                        ]
                      : [
                          { id: 'draft', label: 'Draft ≈30 s' },
                          { id: 'final', label: 'Final ≈6 min' },
                        ]
                  }
                />
              </Field>
            </div>
            {P?.draft && !final && (
              <span className="text-[10px] text-fg3 leading-snug">
                Each draft is its final's first pass (same seed and composition) at about half the time. Pick one, then Upscale it from
                its record; drafts are kept 24 h.
              </span>
            )}
          </>
        )}
      </div>
      <footer className="flex items-center gap-2 p-2 border-t border-line shrink-0">
        <div className="flex-1 min-w-0 text-[11px] leading-tight">
          <SpendLine missing={missing} estErr={estErr} est={est} />
        </div>
        <CountPick value={count} onChange={setCount} />
        {overCap ? (
          <Btn tone="danger" disabled={busy} onClick={() => go(true)} title="Render even though it takes today past the daily cap">
            Past the cap — render anyway
          </Btn>
        ) : (
          <Btn
            tone="deck"
            disabled={!args || busy || !!missing}
            onClick={() => (wouldExceed ? setOverCap(true) : go(false))}
            title="Start the jobs; they land in the folder on the “Renders land in” card"
          >
            {busy ? 'Starting…' : count > 1 ? `Render ${count}` : 'Render'}
          </Btn>
        )}
      </footer>
    </div>
  );
}

/** Modal | Local with a state lamp each; under it one line on the chosen
 *  backend's state, and Start while the desktop's job service is down. */
function BackendPick({ value, onChange }: { value: TfBackend; onChange: (b: TfBackend) => void }) {
  const backends = useQueue((s) => s.backends);
  const unconfigured = backends?.local.state === 'unconfigured';
  // A machine without a desktop GPU never sends to Local.
  useEffect(() => {
    if (unconfigured && value === 'local') onChange('modal');
  }, [unconfigured, value]);
  const look = backendLook(value, backends);
  const [busy, setBusy] = useState(false);
  return (
    <Group label="Backend">
      <div className="inline-flex self-start rounded-md border border-line bg-panel2 p-[2px]">
        {BACKENDS.map((b) => {
          const l = backendLook(b, backends);
          const off = b === 'local' && unconfigured;
          return (
            <button
              key={b}
              type="button"
              disabled={off}
              onClick={() => onChange(b)}
              title={`${BACKEND_LABEL[b]} — ${l.line}`}
              className={
                'flex items-center gap-1.5 h-[22px] px-2.5 rounded-[4px] text-[11px] font-medium transition-colors ' +
                (value === b ? 'bg-raised text-fg shadow-inset' : 'text-fg3 hover:text-fg') +
                (off ? ' opacity-40 pointer-events-none' : '')
              }
            >
              <Dot tone={l.tone} />
              {BACKEND_LABEL[b]}
              <span className="text-[10px] text-fg3 font-normal">{l.word}</span>
            </button>
          );
        })}
      </div>
      <div className="flex items-start gap-2">
        <span className="text-[10px] text-fg3 leading-snug flex-1">
          {look.line}
        </span>
        {value === 'local' && backends?.local.state === 'service_down' && (
          <Btn
            size="xs"
            tone="deck"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              await startLocal();
              setBusy(false);
            }}
            title="Start the desktop's job service (textureflow-ctl start over SSH)"
          >
            Start
          </Btn>
        )}
      </div>
      {value === 'modal' && <ModalSlots />}
    </Group>
  );
}

const images = (paths: string[]) => paths.filter(isImagePath);

/** A Render panel slot that takes library images dragged from the grid
 *  (pointer drag, `ui/media/pathDrag.ts`); it lights up while one hovers. */
function DropSlot({ onPaths, accept, children }: { onPaths: (p: string[]) => void; accept: (p: string[]) => string[]; children: React.ReactNode }) {
  const { dropProps, hovering } = usePathDrop(onPaths, accept);
  return (
    <div {...dropProps} className={'rounded ' + (hovering ? 'ring-1 ring-deck bg-deck/10' : '')}>
      {children}
    </div>
  );
}

function RecipeHint({ recipe }: { recipe?: Recipe }) {
  if (!recipe) return null;
  return (
    <div className="flex gap-2 mt-1.5">
      {recipe.preview && <img src={mediaUrl(recipe.preview)} className="w-14 h-14 object-cover rounded border border-line shrink-0" />}
      <span className="text-[10px] text-fg3 leading-snug">{recipe.when_to_use}</span>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      {children}
    </label>
  );
}

/** `Field` without the <label>: for controls holding buttons / faders, where
 *  a label would forward every click to its first button (the picker). */
function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      {children}
    </div>
  );
}

/** Under a slot: "auto · <region>" while it holds the region's crop, else a
 *  link back to it. Nothing without a region. */
function AutoHint({ auto, value, onUse, region }: { auto?: string | null; value: string | null; onUse: (p: string) => void; region?: string }) {
  if (!auto || !region) return null;
  return value === auto ? (
    <span className="text-[10px] text-fg3" title="Cut from the region's box — follows the Made for choice">
      auto · {region}
    </span>
  ) : (
    <button className="text-[10px] text-fg3 hover:text-fg underline-offset-2 hover:underline text-left" onClick={() => onUse(auto)}>
      use {region}'s crop
    </button>
  );
}

/** How many renders to start: a small "×N" chip by the Render button (1 by
 *  default — batches are rare), opening a 1–12 grid. Not a native <select>:
 *  macOS draws its menu light. */
function CountPick({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  const [anchor, setAnchor] = useState<HTMLButtonElement | null>(null);
  return (
    <>
      <button
        onClick={(e) => setAnchor(anchor ? null : e.currentTarget)}
        className={'h-7 px-1.5 rounded border border-line text-[11px] num shrink-0 hover:text-fg ' + (value > 1 ? 'text-fg' : 'text-fg3')}
        title="How many renders to start — separate jobs, each with a random seed unless Seed is set"
      >
        ×{value} ▾
      </button>
      {anchor && (
        <Popover anchor={anchor} onClose={() => setAnchor(null)} width={148}>
          <div className="grid grid-cols-4 gap-1 p-1">
            {Array.from({ length: 12 }, (_, i) => i + 1).map((n) => (
              <button
                key={n}
                onClick={() => {
                  onChange(n);
                  setAnchor(null);
                }}
                className={'h-7 rounded text-[12px] num ' + (n === value ? 'bg-deck/25 text-fg' : 'text-fg2 hover:bg-panel2 hover:text-fg')}
              >
                {n}
              </button>
            ))}
          </div>
        </Popover>
      )}
    </>
  );
}

/** A slot with a small × that empties it (only while it holds something). */
function Clearable({ value, onClear, children }: { value: string | null; onClear: () => void; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1 min-w-0">
      <div className="flex-1 min-w-0">{children}</div>
      {value && (
        <button
          onClick={onClear}
          className="w-6 h-7 shrink-0 rounded text-fg3 hover:text-fg hover:bg-panel2 text-[14px] leading-none"
          title="Clear"
          aria-label="Clear"
        >
          ×
        </button>
      )}
    </div>
  );
}
