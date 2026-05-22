// Corner preview thumbnail — confirms the projector window is alive without
// alt-tabbing. Reads the `preview` telemetry channel (JPEG bytes b64-encoded
// at ~15fps) and pipes the data: URL straight into an <img>.
//
// Size variants: 'corner' for the Prepare overview (320×~), 'fill' for the
// Perform route hero shot.

import { useStore } from '../state/store';

export function PreviewThumbnail({
  variant = 'corner',
}: {
  variant?: 'corner' | 'fill';
}) {
  const preview = useStore((s) => s.preview);
  const pack = useStore((s) => s.pack);

  const aspect = pack ? `${pack.width} / ${pack.height}` : '16 / 9';
  const src = preview ? `data:image/jpeg;base64,${preview.data_b64}` : null;

  return (
    <div
      className={
        'rounded border border-ink-600 bg-ink-900 overflow-hidden ' +
        (variant === 'corner' ? 'w-[320px]' : 'w-full')
      }
      style={{ aspectRatio: aspect }}
    >
      {src ? (
        <img
          src={src}
          alt="projector preview"
          className="w-full h-full object-cover"
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center text-xs text-zinc-500">
          waiting for first preview frame…
        </div>
      )}
    </div>
  );
}
