// Surface shine-through: one slider per preview view (Prepare, Perform's
// DESIGN, Perform's LIVE), each remembered per machine. The engine turns the
// amount into a physical approximation of the light on the real wall
// (`PreviewParams.surface` → `projected_on_surface` in final_pass.wgsl):
// 0 = pure light on a black wall, 1 = the light reflected off the
// photographed wall plus the wall under room light.

import { useState } from 'react';
import { Fader } from './Fader';

export type SurfaceView = 'prepare' | 'design' | 'live';

const KEY = (view: SurfaceView) => `wzrd.surface.${view}`;
const DEFAULT: Record<SurfaceView, number> = { prepare: 0.5, design: 0, live: 0 };

function read(view: SurfaceView): number {
  try {
    const v = Number(localStorage.getItem(KEY(view)));
    return localStorage.getItem(KEY(view)) != null && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : DEFAULT[view];
  } catch {
    return DEFAULT[view];
  }
}

/** The view's amount + its setter (persisted, try/catch: storage can be blocked). */
export function useSurfaceAmount(view: SurfaceView): [number, (v: number) => void] {
  const [values, setValues] = useState<Partial<Record<SurfaceView, number>>>({});
  const value = values[view] ?? read(view);
  const set = (v: number) => {
    setValues((s) => ({ ...s, [view]: v }));
    try {
      localStorage.setItem(KEY(view), String(v));
    } catch {
      /* storage blocked — the slider still works this session */
    }
  };
  return [value, set];
}

export function SurfaceSlider({ view, value, onChange, className = '' }: { view: SurfaceView; value: number; onChange: (v: number) => void; className?: string }) {
  return (
    <Fader
      value={value}
      min={0}
      max={1}
      step={0.01}
      defaultValue={DEFAULT[view]}
      onChange={onChange}
      label="surface"
      format={(v) => (v > 0 ? `${Math.round(v * 100)}%` : 'off')}
      size="sm"
      className={className}
      title={
        'How much the real wall shows through, approximating projection on it: 0 = pure light on a black wall; ' +
        'higher = the light reflected off the photographed surface (dark cracks swallow it, pale stone throws it back) ' +
        'plus the wall under room light. Double-click: default. Each view keeps its own value.'
      }
    />
  );
}
