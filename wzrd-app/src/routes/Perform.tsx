// Perform route — preview hero (top), audio feature strip (middle), driver
// rack (bottom). Tuned for showtime glanceability; everything you adjust at
// the laptop during a set lives in the driver rack.

import { PreviewThumbnail } from '../components/PreviewThumbnail';
import { AudioStrip } from '../components/AudioStrip';
import { DriverRack } from '../components/DriverRack';

export function Perform() {
  return (
    <div className="grid grid-rows-[auto_auto_1fr] h-full min-h-0">
      <section className="border-b border-ink-700 p-4 flex justify-center">
        <div className="max-w-[60%]">
          <PreviewThumbnail variant="fill" />
        </div>
      </section>
      <section className="border-b border-ink-700 p-4">
        <AudioStrip />
      </section>
      <section className="min-h-0 overflow-auto p-4">
        <DriverRack />
      </section>
    </div>
  );
}
