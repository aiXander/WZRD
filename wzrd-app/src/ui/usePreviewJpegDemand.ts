// Hold a refcounted demand on the JPEG `preview` channel while `active`.
// Without one the app receives no `preview` frames at all (see
// `previewJpegDemand`).

import { useEffect } from 'react';
import { previewJpegDemand } from '../api/ipc';

export function usePreviewJpegDemand(active: boolean) {
  useEffect(() => {
    if (!active) return;
    previewJpegDemand(true).catch(() => {});
    return () => {
      previewJpegDemand(false).catch(() => {});
    };
  }, [active]);
}
