// Hold demand on the JPEG `preview` channel while `active`. Without one the
// app receives no `preview` frames at all (see api/telemetryDemand.ts) —
// and the engine skips the JPEG readback and, in two-leg mode, the design
// leg's render.

import { useTelemetryDemand } from '../api/telemetryDemand';

export function usePreviewJpegDemand(active: boolean) {
  useTelemetryDemand('preview', active);
}
