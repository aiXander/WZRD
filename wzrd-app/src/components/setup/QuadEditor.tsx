// The lit photo with a draggable 4-corner quad (normalised coords, TL TR BR
// BL) — the manual rescue for a failed or sloppy projection detection.
//
// Precision: handles have a large invisible grab radius; ⇧-drag moves at ¼
// speed; a 4× loupe follows the dragged corner; a clicked corner nudges by
// one photo pixel with the arrow keys (⇧ = 10).

import { useEffect, useRef, useState } from 'react';
import { mediaUrl } from '../../api/hub';
import { useAspectFit } from '../../ui/useAspectFit';
import type { Corners } from '../../state/setup';

const LABELS = ['TL', 'TR', 'BR', 'BL'];
const LOUPE = 150;
const ZOOM = 4;

export function QuadEditor({
  path,
  corners,
  onChange,
  height = 420,
  dim = false,
}: {
  path: string;
  corners: Corners | null;
  /** Every move (manual from here on). */
  onChange: (c: Corners) => void;
  height?: number;
  /** Grey the photo out (a run is in flight). */
  dim?: boolean;
}) {
  const [wrap, setWrap] = useState<HTMLDivElement | null>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const box = useAspectFit(wrap, nat ? nat.w / nat.h : 4 / 3);
  const [drag, setDrag] = useState<number | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const start = useRef<{ px: number; py: number; c: [number, number] } | null>(null);
  const url = mediaUrl(path);

  useEffect(() => {
    setNat(null);
  }, [path]);

  const pts = corners?.map(([x, y]) => [x * box.w, y * box.h] as [number, number]) ?? null;

  const move = (i: number, x: number, y: number) => {
    if (!corners) return;
    const c = corners.map((p, k) => (k === i ? ([clamp01(x), clamp01(y)] as [number, number]) : p)) as Corners;
    onChange(c);
  };

  const onDown = (i: number) => (e: React.PointerEvent) => {
    if (!corners) return;
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    start.current = { px: e.clientX, py: e.clientY, c: corners[i] };
    setDrag(i);
    setSel(i);
  };
  const onMove = (e: React.PointerEvent) => {
    if (drag == null || !start.current || !box.w) return;
    const k = e.shiftKey ? 0.25 : 1;
    const dx = ((e.clientX - start.current.px) * k) / box.w;
    const dy = ((e.clientY - start.current.py) * k) / box.h;
    move(drag, start.current.c[0] + dx, start.current.c[1] + dy);
  };
  const onUp = () => {
    setDrag(null);
    start.current = null;
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (sel == null || !corners || !nat) return;
    const step = (e.shiftKey ? 10 : 1) / Math.max(nat.w, nat.h);
    const d: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const v = d[e.key];
    if (!v) return;
    e.preventDefault();
    e.stopPropagation();
    move(sel, corners[sel][0] + v[0], corners[sel][1] + v[1]);
  };

  const loupeAt = drag ?? null;
  const lp = loupeAt != null && corners ? corners[loupeAt] : null;

  return (
    <div ref={setWrap} className="relative w-full flex items-center justify-center" style={{ height }}>
      <div
        tabIndex={0}
        onKeyDown={onKey}
        className="relative outline-none rounded-md overflow-hidden border border-line bg-black focus-visible:border-deck"
        style={{ width: box.w, height: box.h }}
      >
        <img
          src={url}
          draggable={false}
          onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          className={'absolute inset-0 w-full h-full select-none transition-opacity ' + (dim ? 'opacity-40' : '')}
        />
        {pts && (
          <svg
            className="absolute inset-0"
            width={box.w}
            height={box.h}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerCancel={onUp}
          >
            <polygon
              points={pts.map((p) => p.join(',')).join(' ')}
              fill="rgb(var(--deck) / 0.12)"
              stroke="rgb(var(--deck))"
              strokeWidth={1.5}
            />
            {pts.map(([x, y], i) => (
              <g key={i} onPointerDown={onDown(i)} style={{ cursor: drag === i ? 'grabbing' : 'grab' }}>
                <circle cx={x} cy={y} r={18} fill="transparent" />
                <circle
                  cx={x}
                  cy={y}
                  r={7}
                  fill={sel === i ? 'rgb(var(--deck))' : 'rgb(0 0 0 / 0.55)'}
                  stroke="rgb(var(--deck))"
                  strokeWidth={2}
                />
                <circle cx={x} cy={y} r={1.5} fill="white" />
                <text
                  x={x + (corners![i][0] < 0.5 ? 12 : -12)}
                  y={y + (corners![i][1] < 0.5 ? 20 : -12)}
                  textAnchor={corners![i][0] < 0.5 ? 'start' : 'end'}
                  className="fill-white text-[10px] font-semibold"
                  style={{ paintOrder: 'stroke', stroke: 'rgba(0,0,0,.7)', strokeWidth: 3 }}
                >
                  {LABELS[i]}
                </text>
              </g>
            ))}
          </svg>
        )}
        {lp && (
          <div
            className="absolute pointer-events-none rounded-full border-2 border-deck shadow-[0_8px_24px_rgba(0,0,0,.6)] overflow-hidden"
            style={{
              width: LOUPE,
              height: LOUPE,
              top: lp[1] > 0.5 ? 10 : undefined,
              bottom: lp[1] > 0.5 ? undefined : 10,
              left: lp[0] > 0.5 ? 10 : undefined,
              right: lp[0] > 0.5 ? undefined : 10,
              backgroundColor: 'black',
              backgroundImage: `url("${url}")`,
              backgroundRepeat: 'no-repeat',
              backgroundSize: `${box.w * ZOOM}px ${box.h * ZOOM}px`,
              backgroundPosition: `${LOUPE / 2 - lp[0] * box.w * ZOOM}px ${LOUPE / 2 - lp[1] * box.h * ZOOM}px`,
            }}
          >
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-deck/80" />
            <div className="absolute top-1/2 left-0 right-0 h-px bg-deck/80" />
          </div>
        )}
      </div>
    </div>
  );
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** A starting quad for drawing by hand: 10 % in from each edge. */
export const DEFAULT_QUAD: Corners = [
  [0.1, 0.1],
  [0.9, 0.1],
  [0.9, 0.9],
  [0.1, 0.9],
];
