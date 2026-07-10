// Live preview of the composite buffer — what the projector is painting.
// Reads the `preview` telemetry channel (JPEG bytes b64-encoded at ~15fps)
// and pipes the data: URL straight into an <img>.
//
// Size variants: 'corner' for the Prepare overview (320×~), 'fill' scales
// to whatever space the parent gives it (Perform hero) while preserving
// the pack aspect ratio.

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

  if (variant === 'fill') {
    return (
      <div className="w-full h-full flex items-center justify-center">
        {src ? (
          <img
            src={src}
            alt="projector preview"
            className="max-w-full max-h-full object-contain rounded border border-ink-600"
            style={{ aspectRatio: aspect }}
          />
        ) : (
          <div
            className="h-full max-w-full rounded border border-ink-600 bg-ink-900 flex items-center justify-center text-xs text-zinc-500"
            style={{ aspectRatio: aspect }}
          >
            waiting for first preview frame…
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className="rounded border border-ink-600 bg-ink-900 overflow-hidden w-[320px]"
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
