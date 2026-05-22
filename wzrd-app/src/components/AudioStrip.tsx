// Audio feature strip — three vertical bars for `audio.band(low/mid/high)`
// + three onset flashes with their current decay envelopes.
//
// This is the post-refactor audio-debug viz (per `audio_refactor_plan.md`
// §10). Tuning (gates, compression, onset thresholds, BPM smoothing) lives
// on the audio server's browser UI — clicking the OSC pill in the status
// strip opens it. WZRD's job is *routing*, not DSP.

import { useStore } from '../state/store';

function bar(value: number, label: string, accent: string) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="w-8 h-32 bg-ink-700 rounded overflow-hidden flex flex-col-reverse">
        <div
          className={accent + ' w-full transition-[height] duration-75'}
          style={{ height: `${pct}%` }}
        />
      </div>
      <div className="text-[10px] text-zinc-400">{label}</div>
      <div className="text-[10px] text-zinc-300 font-mono">{value.toFixed(2)}</div>
    </div>
  );
}

function onsetFlash(value: number, label: string, accent: string) {
  return (
    <div className="flex flex-col items-center gap-1 w-12">
      <div
        className={
          'h-12 w-12 rounded-full border ' +
          (value > 0.05
            ? accent + ' border-transparent'
            : 'border-ink-600 bg-ink-800')
        }
        style={{ opacity: Math.max(0.15, Math.min(1, value)) }}
      />
      <div className="text-[10px] text-zinc-400">{label}</div>
      <div className="text-[10px] text-zinc-300 font-mono">{value.toFixed(2)}</div>
    </div>
  );
}

export function AudioStrip() {
  const audio = useStore((s) => s.audio);
  if (!audio) {
    return (
      <div className="text-xs text-zinc-500">
        waiting for audio… (is the audio-feature-server running?)
      </div>
    );
  }
  return (
    <div className="flex items-end gap-8">
      <div className="flex items-end gap-4">
        {bar(audio.band_low, 'low', 'bg-accent-blue')}
        {bar(audio.band_mid, 'mid', 'bg-accent-violet')}
        {bar(audio.band_high, 'high', 'bg-accent-amber')}
      </div>
      <div className="text-zinc-500 text-xs self-center">onsets</div>
      <div className="flex items-end gap-3">
        {onsetFlash(audio.onset_low, 'low', 'bg-accent-blue')}
        {onsetFlash(audio.onset_mid, 'mid', 'bg-accent-violet')}
        {onsetFlash(audio.onset_high, 'high', 'bg-accent-amber')}
      </div>
    </div>
  );
}
