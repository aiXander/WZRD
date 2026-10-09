import { useLayoutEffect, useState } from 'react';

/** Largest `aspect` (w/h) box that fits inside `el`, tracked on resize. */
export function useAspectFit(el: HTMLElement | null, aspect: number) {
  const [box, setBox] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    if (!el) return;
    const fit = () => {
      const r = el.getBoundingClientRect();
      let w = r.width;
      let h = w / aspect;
      if (h > r.height) {
        h = r.height;
        w = h * aspect;
      }
      setBox({ w: Math.floor(w), h: Math.floor(h) });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el, aspect]);
  return box;
}
