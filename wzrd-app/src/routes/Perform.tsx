// Perform route — preview hero (fills available height), audio feature
// strip, driver rack. Tuned for showtime glanceability; everything you
// adjust at the laptop during a set lives in the driver rack.

import { PreviewThumbnail } from '../components/PreviewThumbnail';
import { AudioStrip } from '../components/AudioStrip';
import { DriverRack } from '../components/DriverRack';

export function Perform() {
  return (
    <div className="grid grid-rows-[minmax(0,1fr)_auto_minmax(0,45%)] h-full min-h-0">
      <section className="border-b border-ink-700 p-3 min-h-0">
        <PreviewThumbnail variant="fill" />
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
