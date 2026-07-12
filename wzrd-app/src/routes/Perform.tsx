// Perform route — preview hero (fills available height), masters row
// (§5.4, always visible), audio feature strip, driver rack. Tuned for
// showtime glanceability; everything you adjust at the laptop during a set
// lives in the masters row + driver rack.

import { NativePreview } from '../components/NativePreview';
import { MastersRow } from '../components/MastersRow';
import { AudioStrip } from '../components/AudioStrip';
import { DriverRack } from '../components/DriverRack';

export function Perform() {
  return (
    <div className="grid grid-rows-[minmax(0,1fr)_auto_auto_minmax(0,45%)] h-full min-h-0">
      <section className="border-b border-ink-700 p-3 min-h-0">
        {/* Collapse Step 3: hero is the native composite blit — lossless,
            full-rate — not the JPEG thumbnail. */}
        <NativePreview />
      </section>
      <section className="border-b border-ink-700 px-4 py-2">
        <MastersRow />
      </section>
      <section className="border-b border-ink-700 px-4 py-2">
        <AudioStrip />
      </section>
      <section className="min-h-0 overflow-auto p-4">
        <DriverRack />
      </section>
    </div>
  );
}
