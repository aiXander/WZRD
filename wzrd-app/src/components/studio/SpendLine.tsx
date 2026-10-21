// The "≈ $x · today $y of $cap" line under every render button (Studio's
// Render panel, the setup wizard's Segmentation step), and the over-cap test
// that decides whether Go turns into "Past the cap — render anyway".

import type { Estimate } from '../../api/hub';

export const wouldExceedCap = (est: Estimate | null) =>
  !!est && est.spend.today_usd + est.spend.running_usd + est.usd > est.spend.cap_usd;

export function SpendLine({ missing, estErr, est }: { missing?: string | null; estErr?: string | null; est: Estimate | null }) {
  if (missing) return <span className="text-fg3">{missing}</span>;
  if (estErr) return <span className="text-bad">{estErr}</span>;
  if (!est) return <span className="text-fg3">estimating…</span>;
  return (
    <span className={wouldExceedCap(est) ? 'text-bad' : est.spend.warn ? 'text-amber' : 'text-fg2'}>
      ≈ ${est.usd.toFixed(2)} · today ${est.spend.today_usd.toFixed(2)} of ${est.spend.cap_usd.toFixed(0)}
    </span>
  );
}
