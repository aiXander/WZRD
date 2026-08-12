// §5.14 alignment canvas — drag the light onto the wall.
//
// The view is *dest* space padded ~20% on every side, because handles are
// routinely dragged off-screen: the projector rectangle sits in the middle
// with room to pull a corner outward. Coordinates on the wire are normalized
// dest uv; the SVG works in "canvas units" that equal projector pixels, so a
// keyboard nudge can be exactly one output pixel.
//
// Ground truth is always the projector, but the canvas shows the real field:
// a source-space grid drawn where it actually lands (`warpMath`, evaluated
// from the engine's own solved coefficients), so a local correction is
// visible here the moment you drag it. The photographic underlay behind it is
// the composite preview through the **corner quad only** — a homography maps
// exactly onto a CSS `matrix3d`, so it comes for free, but CSS cannot express
// the local corrections. Grid = truth, image = reference.

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { AlignmentDoc, WarpPoint } from '../api/ipc';
import { useStore } from '../state/store';
import {
  addPoint,
  flushAlignment,
  panCorners,
  removePoint,
  setCorner,
  setPointDest,
  setPointRadius,
} from '../state/alignment';
import { boundarySamples, makeWarp, snapToBoundary, warpGrid } from '../state/warpMath';

/** Padding around the projector rect, as a fraction of its size. */
const PAD = 0.2;
/** Fallback output size before the engine reports one. */
const FALLBACK: [number, number] = [1920, 1080];

const CORNER_LABELS = ['TL', 'TR', 'BR', 'BL'];
/** Handle key → corner index; `undefined` for a point id. */
const CORNER_INDEX: Record<string, number | undefined> = { c0: 0, c1: 1, c2: 2, c3: 3 };
const ARROWS: Record<string, [number, number] | undefined> = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
};

type Menu = { x: number; y: number; target: { kind: 'corner'; index: number } | { kind: 'point'; id: string } | { kind: 'empty'; dest: [number, number] } };

type Handle =
  | { kind: 'corner'; index: number }
  | { kind: 'point'; id: string };

/**
 * An in-flight gesture. Panning carries the corner set it started from and
 * the grab point, so the drag stays absolute instead of integrating deltas
 * (which drifts once the engine's echo lands mid-gesture).
 */
type Drag =
  | { kind: 'handle'; handle: Handle }
  | { kind: 'pan'; from: [number, number][]; grab: [number, number] }
  /**
   * An edge click that is becoming a handle. The engine names the handle (and
   * anchors it), so there's a round trip before there is anything to drag;
   * this state absorbs the pointer moves in between and upgrades to `handle`
   * when the id lands — pressing and dragging in one motion works, which is
   * how you'd expect to place an edge point.
   */
  | { kind: 'adding' };

function handleKey(h: Handle) {
  return h.kind === 'corner' ? `c${h.index}` : h.id;
}

/**
 * One extra handle: a dashed circle for its support radius (how far this
 * correction reaches — a radius that swallows a corner *will* move that
 * corner, and you want to see that here rather than discover it on the wall),
 * a tether back to its content anchor, and the grab dot itself.
 *
 * `px` converts CSS pixels to canvas units, so handle dots stay a constant
 * on-screen size no matter how the view is fitted.
 */
function PointHandle({
  point,
  out,
  px,
  selected,
  onGrab,
  onMenu,
}: {
  point: WarpPoint;
  out: [number, number];
  px: (n: number) => number;
  selected: boolean;
  onGrab: (e: React.PointerEvent) => void;
  onMenu: (e: React.MouseEvent) => void;
}) {
  const [w, h] = out;
  const [cx, cy] = [point.dest[0] * w, point.dest[1] * h];
  return (
    <g>
      <circle
        cx={cx}
        cy={cy}
        r={point.radius * Math.min(w, h)}
        fill="none"
        stroke={selected ? '#f59e0b' : '#52525b'}
        strokeDasharray="6 8"
        vectorEffect="non-scaling-stroke"
        strokeWidth={1}
        pointerEvents="none"
      />
      <line
        x1={point.anchor[0] * w}
        y1={point.anchor[1] * h}
        x2={cx}
        y2={cy}
        stroke="#f59e0b"
        strokeOpacity={0.5}
        vectorEffect="non-scaling-stroke"
        strokeWidth={1}
        pointerEvents="none"
      />
      <circle
        cx={cx}
        cy={cy}
        r={px(9)}
        fill={selected ? '#f59e0b' : '#fbbf24'}
        stroke="#18181b"
        vectorEffect="non-scaling-stroke"
        strokeWidth={1.5}
        className="cursor-grab"
        onPointerDown={onGrab}
        onContextMenu={onMenu}
      />
    </g>
  );
}

function CornerHandle({
  at,
  label,
  px,
  selected,
  onGrab,
  onMenu,
}: {
  at: [number, number];
  label: string;
  px: (n: number) => number;
  selected: boolean;
  onGrab: (e: React.PointerEvent) => void;
  onMenu: (e: React.MouseEvent) => void;
}) {
  const r = px(11);
  return (
    <g>
      <circle
        cx={at[0]}
        cy={at[1]}
        r={r}
        fill={selected ? '#38bdf8' : '#0ea5e9'}
        stroke="#18181b"
        vectorEffect="non-scaling-stroke"
        strokeWidth={1.5}
        className="cursor-grab"
        onPointerDown={onGrab}
        onContextMenu={onMenu}
      />
      <text
        x={at[0]}
        y={at[1] - r * 1.6}
        textAnchor="middle"
        fill="#71717a"
        style={{ fontSize: `${px(14)}px` }}
        pointerEvents="none"
      >
        {label}
      </text>
    </g>
  );
}

const IDENTITY_CORNERS: [number, number][] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
];

/**
 * Right-click menu. Split out of the canvas so each branch narrows the target
 * once, at the top, instead of re-proving it inside every callback.
 *
 * Corners get **Reset corner** and nothing else: they are structural, and
 * removing one would destroy the projective base the whole warp stands on.
 */
function HandleMenu({
  menu,
  doc,
  onClose,
}: {
  menu: Menu;
  doc: AlignmentDoc;
  onClose: () => void;
}) {
  const cls = 'block w-full px-3 py-1.5 text-left text-zinc-200 hover:bg-ink-700';
  const t = menu.target;
  return (
    <div
      className="absolute z-10 min-w-[13rem] rounded border border-ink-600 bg-ink-800 py-1 text-xs shadow-lg"
      style={{ left: menu.x, top: menu.y }}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {t.kind === 'empty' && (
        <button
          className={cls}
          onClick={() => {
            void addPoint(t.dest);
            onClose();
          }}
        >
          Add point here
        </button>
      )}
      {t.kind === 'point' && (
        <>
          <button
            className={cls}
            onClick={() => {
              removePoint(t.id);
              onClose();
            }}
          >
            Remove point
          </button>
          <div className="mt-1 border-t border-ink-700 px-3 pb-1 pt-2">
            <label className="block text-[10px] uppercase tracking-wide text-zinc-500">
              Radius — how far this correction reaches
            </label>
            <input
              type="range"
              min={0.02}
              max={1.2}
              step={0.01}
              className="w-full"
              defaultValue={doc.points.find((p) => p.id === t.id)?.radius ?? 0.35}
              onChange={(e) => setPointRadius(t.id, Number(e.target.value))}
            />
          </div>
        </>
      )}
      {t.kind === 'corner' && (
        <button
          className={cls}
          onClick={() => {
            setCorner(t.index, IDENTITY_CORNERS[t.index]);
            void flushAlignment();
            onClose();
          }}
        >
          Reset corner
        </button>
      )}
    </div>
  );
}

export function WarpCanvas() {
  const doc = useStore((s) => s.alignment);
  const preview = useStore((s) => s.preview);
  const selected = useStore((s) => s.selectedHandle);
  const setSelected = useStore((s) => s.setSelectedHandle);

  const hostRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<Drag | null>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [menu, setMenu] = useState<Menu | null>(null);
  const [panning, setPanning] = useState(false);
  /** Snapped position under the pointer while it hovers the quad outline. */
  const [edgeHover, setEdgeHover] = useState<[number, number] | null>(null);

  // The SVG scales itself with preserveAspectRatio; the underlay <img> is a
  // plain DOM element, so the same fit has to be reproduced in JS to place it.
  useLayoutEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setBox({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setBox({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const [outW, outH] = doc && doc.output[0] > 0 ? doc.output : FALLBACK;
  const vb = {
    x: -PAD * outW,
    y: -PAD * outH,
    w: (1 + 2 * PAD) * outW,
    h: (1 + 2 * PAD) * outH,
  };
  // uv → canvas units (= projector px), then canvas units → CSS px.
  const fit = box.w > 0 && box.h > 0 ? Math.min(box.w / vb.w, box.h / vb.h) : 0;
  const originX = fit > 0 ? (box.w - vb.w * fit) / 2 - vb.x * fit : 0;
  const originY = fit > 0 ? (box.h - vb.h * fit) / 2 - vb.y * fit : 0;

  /** Client px → dest uv, via the SVG's own CTM so the fit math can't drift. */
  const toUv = useCallback((clientX: number, clientY: number): [number, number] => {
    const svg = svgRef.current;
    if (!svg) return [0, 0];
    const ctm = svg.getScreenCTM();
    if (!ctm) return [0, 0];
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const p = pt.matrixTransform(ctm.inverse());
    return [p.x / outW, p.y / outH];
  }, [outW, outH]);

  const beginDrag = (e: React.PointerEvent, h: Handle) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = { kind: 'handle', handle: h };
    setSelected(handleKey(h));
    setMenu(null);
  };

  /** Drag the quad body: move the whole image on the wall, no reshaping. */
  const beginPan = (e: React.PointerEvent) => {
    if (e.button !== 0 || !doc) return;
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = {
      kind: 'pan',
      from: doc.corners.map((c) => [...c] as [number, number]),
      grab: toUv(e.clientX, e.clientY),
    };
    setSelected(null);
    setPanning(true);
    setMenu(null);
  };

  /**
   * Press on the quad outline: drop a handle *on the edge* and start dragging
   * it in the same motion.
   *
   * These are ordinary RBF handles — they are not a fifth, sixth… corner (only
   * four points can define the projective base). What they give you is a
   * control point that starts out exactly on the edge, so pulling it bends
   * that side to follow a wall that isn't straight. Reach is the handle's
   * radius, same as any other; shrink it for a local dent, widen it to bow the
   * whole side.
   */
  const beginEdgeAdd = (e: React.PointerEvent) => {
    if (e.button !== 0 || !doc || !edgeHover) return;
    e.preventDefault();
    e.stopPropagation();
    const el = e.target as Element;
    const pid = e.pointerId;
    el.setPointerCapture(pid);
    dragRef.current = { kind: 'adding' };
    setMenu(null);
    const at = edgeHover;
    setEdgeHover(null); // the real handle is about to take its place
    void addPoint(at).then((id) => {
      if (!id) {
        // Rejected (handle cap) — the error pill says why.
        if (dragRef.current?.kind === 'adding') dragRef.current = null;
        return;
      }
      setSelected(id);
      // Only take over the gesture if the pointer is still down on it.
      if (dragRef.current?.kind === 'adding') {
        dragRef.current = { kind: 'handle', handle: { kind: 'point', id } };
      }
    });
  };

  const onMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    if (d.kind === 'adding') return; // waiting on the engine to name it
    const uv = toUv(e.clientX, e.clientY);
    if (d.kind === 'pan') {
      panCorners(d.from, [uv[0] - d.grab[0], uv[1] - d.grab[1]]);
    } else if (d.handle.kind === 'corner') {
      setCorner(d.handle.index, uv);
    } else {
      setPointDest(d.handle.id, uv);
    }
  };

  const endDrag = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    setPanning(false);
    (e.target as Element).releasePointerCapture?.(e.pointerId);
    // Don't leave the last position of a drag queued behind a frame that may
    // never come (tab switch, window hide).
    void flushAlignment();
  };

  // Arrow keys nudge by exactly one output pixel (shift = 10) — the last
  // millimetre of physical alignment, and the one thing a mouse genuinely
  // cannot do. With a handle selected it moves that handle; with nothing
  // selected it pans the whole quad, which is the same gesture at the same
  // precision for "the image is two pixels left of the wall".
  useEffect(() => {
    if (!doc) return;
    function onKey(e: KeyboardEvent) {
      const dir = ARROWS[e.key];
      if (!dir) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      e.preventDefault();
      const step = e.shiftKey ? 10 : 1;
      const du = (dir[0] * step) / outW;
      const dv = (dir[1] * step) / outH;
      const d = doc!;
      const ci = selected ? CORNER_INDEX[selected] : undefined;
      if (!selected) {
        panCorners(d.corners, [du, dv]);
      } else if (ci !== undefined) {
        const c = d.corners[ci];
        setCorner(ci, [c[0] + du, c[1] + dv]);
      } else {
        const p = d.points.find((q) => q.id === selected);
        if (p) setPointDest(p.id, [p.dest[0] + du, p.dest[1] + dv]);
      }
      void flushAlignment();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected, doc, outW, outH]);

  // The field itself, drawn as a source-space grid mapped to where it lands.
  // This is what makes a local correction visible: the photographic underlay
  // below can only express the corner homography (CSS has no such thing as a
  // non-projective transform), so without this the operator drags a handle
  // and nothing in the UI moves.
  const warp = useMemo(() => (doc ? makeWarp(doc) : null), [doc]);
  const grid = useMemo(
    () => (warp ? warpGrid(warp, 8, 16, [outW, outH]) : []),
    [warp, outW, outH]
  );
  // Sampled once per document so hovering the edge stays cheap.
  const boundary = useMemo(
    () => (warp ? boundarySamples(warp, 48) : []),
    [warp]
  );

  if (!doc) {
    return (
      <div className="h-full grid place-items-center text-xs text-zinc-500">
        waiting for the engine's alignment document…
      </div>
    );
  }

  const corners = doc.corners;
  const quad = corners.map(([u, v]) => `${u * outW},${v * outH}`).join(' ');
  const hom = warp?.h ?? null;
  // CSS px → canvas units, so handle dots keep a constant on-screen size
  // however the view happens to be fitted.
  const px = (n: number) => Math.max(n * 0.6, n / Math.max(fit, 1e-6));

  // Underlay: source unit square → dest quad, expressed in CSS px. The image
  // element's own space is uv scaled by (outW*fit, outH*fit), so conjugate the
  // homography by that scale before packing it into matrix3d (column-major).
  let underlay: React.CSSProperties | null = null;
  if (hom && fit > 0 && preview) {
    const sx = outW * fit;
    const sy = outH * fit;
    const [a, b, c, d2, e2, f, g, h2, i] = hom;
    const m = [a, (b * sx) / sy, c * sx, (d2 * sy) / sx, e2, f * sy, g / sx, h2 / sy, i];
    underlay = {
      position: 'absolute',
      left: originX,
      top: originY,
      width: sx,
      height: sy,
      transformOrigin: '0 0',
      transform: `matrix3d(${m[0]},${m[3]},0,${m[6]},${m[1]},${m[4]},0,${m[7]},0,0,1,0,${m[2]},${m[5]},0,${m[8]})`,
    };
  }

  const openMenu = (e: React.MouseEvent, target: Menu['target']) => {
    e.preventDefault();
    e.stopPropagation();
    const host = hostRef.current?.getBoundingClientRect();
    setMenu({
      x: e.clientX - (host?.left ?? 0),
      y: e.clientY - (host?.top ?? 0),
      target,
    });
  };

  const selectedPoint = doc.points.find((p) => p.id === selected) ?? null;

  return (
    <div ref={hostRef} className="relative h-full w-full overflow-hidden bg-ink-900">
      {underlay && preview && (
        <img
          alt=""
          draggable={false}
          src={`data:image/jpeg;base64,${preview.data_b64}`}
          style={{ ...underlay, opacity: 0.55, pointerEvents: 'none' }}
        />
      )}

      <svg
        ref={svgRef}
        className="absolute inset-0 h-full w-full select-none"
        viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
        onPointerMove={onMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerDown={() => {
          setSelected(null);
          setMenu(null);
        }}
        onContextMenu={(e) => openMenu(e, { kind: 'empty', dest: toUv(e.clientX, e.clientY) })}
      >
        {/* The projector rectangle — where light exists at all. Anything the
            warp pushes outside this is simply not projected. */}
        <rect
          x={0}
          y={0}
          width={outW}
          height={outH}
          fill="none"
          stroke="#3f3f46"
          strokeDasharray="12 10"
          vectorEffect="non-scaling-stroke"
          strokeWidth={1}
        />
        {/* The warped source quad — and the grab target for panning: drag
            anywhere inside it to move the image on the wall without
            reshaping it. */}
        <polygon
          points={quad}
          fill="rgba(56,189,248,0.05)"
          stroke="#38bdf8"
          vectorEffect="non-scaling-stroke"
          strokeWidth={1.5}
          className={panning ? 'cursor-grabbing' : 'cursor-move'}
          onPointerDown={beginPan}
        />

        {/* Edge hit band — invisible, a few CSS px wide, drawn over the
            polygon so the outline wins over panning near the boundary.
            Handles render after it and so still win over both. */}
        <polygon
          points={boundary
            .map(([u, v]) => `${(u * outW).toFixed(2)},${(v * outH).toFixed(2)}`)
            .join(' ')}
          fill="none"
          stroke="transparent"
          strokeWidth={px(14)}
          vectorEffect="non-scaling-stroke"
          pointerEvents="stroke"
          className="cursor-copy"
          onPointerMove={(e) => {
            if (dragRef.current) return;
            setEdgeHover(snapToBoundary(boundary, toUv(e.clientX, e.clientY), [outW, outH]));
          }}
          onPointerLeave={() => setEdgeHover(null)}
          onPointerDown={beginEdgeAdd}
          onContextMenu={(e) => openMenu(e, { kind: 'empty', dest: toUv(e.clientX, e.clientY) })}
        />
        {edgeHover && (
          <g pointerEvents="none">
            <circle
              cx={edgeHover[0] * outW}
              cy={edgeHover[1] * outH}
              r={px(8)}
              fill="none"
              stroke="#fbbf24"
              vectorEffect="non-scaling-stroke"
              strokeWidth={1.5}
            />
            <line
              x1={edgeHover[0] * outW - px(4)}
              y1={edgeHover[1] * outH}
              x2={edgeHover[0] * outW + px(4)}
              y2={edgeHover[1] * outH}
              stroke="#fbbf24"
              vectorEffect="non-scaling-stroke"
              strokeWidth={1.5}
            />
            <line
              x1={edgeHover[0] * outW}
              y1={edgeHover[1] * outH - px(4)}
              x2={edgeHover[0] * outW}
              y2={edgeHover[1] * outH + px(4)}
              stroke="#fbbf24"
              vectorEffect="non-scaling-stroke"
              strokeWidth={1.5}
            />
          </g>
        )}

        {/* The actual field. With no handles this is just the quad
            subdivided; every local correction bends it exactly the way the
            projector will. */}
        {grid.map((line, i) => (
          <polyline
            key={i}
            points={line}
            fill="none"
            stroke="#38bdf8"
            strokeOpacity={warp?.hasResidual ? 0.4 : 0.18}
            vectorEffect="non-scaling-stroke"
            strokeWidth={1}
            pointerEvents="none"
          />
        ))}

        {/* Extra handles: dashed circle = support radius, i.e. how far this
            correction reaches. A radius that swallows a corner will move that
            corner too — visible here before it's visible on the wall. */}
        {doc.points.map((p) => (
          <PointHandle
            key={p.id}
            point={p}
            out={[outW, outH]}
            px={px}
            selected={selected === p.id}
            onGrab={(e) => beginDrag(e, { kind: 'point', id: p.id })}
            onMenu={(e) => openMenu(e, { kind: 'point', id: p.id })}
          />
        ))}

        {/* Corner handles. Structural — there are always exactly four, and
            removing one would destroy the projective base. */}
        {corners.map(([u, v], i) => (
          <CornerHandle
            key={`c${i}`}
            at={[u * outW, v * outH]}
            label={CORNER_LABELS[i]}
            px={px}
            selected={selected === `c${i}`}
            onGrab={(e) => beginDrag(e, { kind: 'corner', index: i })}
            onMenu={(e) => openMenu(e, { kind: 'corner', index: i })}
          />
        ))}
      </svg>

      {menu && (
        <HandleMenu menu={menu} doc={doc} onClose={() => setMenu(null)} />
      )}

      <div className="pointer-events-none absolute bottom-2 left-3 text-[10px] leading-4 text-zinc-500">
        <div>
          drag handles · click the outline to add one on the edge · drag inside to move
          the whole quad · right-click to add/remove · arrows nudge 1 px, ⇧ 10 (nothing
          selected = move the quad) · {outW}×{outH}
        </div>
        <div>
          grid is the real field; the photo underlay only follows the corner quad — ground
          truth is the projector
        </div>
        {selectedPoint && (
          <div className="text-amber-500/80">
            {selectedPoint.id}: dest {(selectedPoint.dest[0] * outW).toFixed(0)},
            {(selectedPoint.dest[1] * outH).toFixed(0)} px · radius{' '}
            {selectedPoint.radius.toFixed(2)}
          </div>
        )}
        {selected?.startsWith('c') && !selectedPoint && (
          <div className="text-sky-400/80">
            {CORNER_LABELS[Number(selected.slice(1))]}:{' '}
            {(corners[Number(selected.slice(1))][0] * outW).toFixed(0)},
            {(corners[Number(selected.slice(1))][1] * outH).toFixed(0)} px
          </div>
        )}
      </div>
    </div>
  );
}
