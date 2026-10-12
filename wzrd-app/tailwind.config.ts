import type { Config } from 'tailwindcss';

// Colours are CSS variables (src/index.css) so the deck mode can re-tint the
// whole room — DESIGN cool, LIVE armed — by flipping one attribute on <html>.
const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: v('bg'),
        panel: v('panel'),
        panel2: v('panel-2'),
        raised: v('raised'),
        line: v('line'),
        line2: v('line-2'),
        fg: v('fg'),
        fg2: v('fg-2'),
        fg3: v('fg-3'),
        fg4: v('fg-4'),
        live: v('live'),
        design: v('design'),
        deck: v('deck'),
        amber: v('amber'),
        mod: v('mod'),
        ok: v('ok'),
        bad: v('bad'),
        low: v('low'),
        mid: v('mid'),
        high: v('high'),
        // Legacy palette — still used by the Align canvas + Debug internals.
        ink: {
          900: '#08090b',
          800: '#0f1115',
          700: '#161922',
          600: '#1f2330',
          500: '#2a2f3e',
          400: '#3a4054',
          300: '#525a73',
        },
        accent: {
          green: '#3ee08f',
          amber: '#ffb02e',
          red: '#ff4a52',
          blue: '#3cc8ff',
          violet: '#b48cff',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"SF Pro Text"',
          'Inter',
          '"Segoe UI"',
          'system-ui',
          'sans-serif',
        ],
        mono: ['"SF Mono"', '"JetBrains Mono"', 'ui-monospace', 'Menlo', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['10px', '12px'],
        xxs: ['9px', '11px'],
      },
      boxShadow: {
        deck: '0 0 0 1px rgb(var(--deck) / 0.55), 0 0 24px -6px rgb(var(--deck) / 0.55)',
        inset: 'inset 0 1px 0 rgb(255 255 255 / 0.04)',
      },
      // Only opacity/transform animate (compositor-only). An infinite
      // box-shadow animation repaints its blur every frame — the armed TAKE
      // button's glow is a pre-shadowed pseudo-element instead (`.arm-glow`
      // in index.css).
      keyframes: {
        fadeIn: { from: { opacity: '0', transform: 'translateY(4px)' }, to: { opacity: '1', transform: 'none' } },
        // Beat lamp: lit on the beat, decays across it (duration = one beat, set inline).
        beatDecay: { from: { opacity: '1' }, to: { opacity: '0.28' } },
      },
      animation: {
        in: 'fadeIn 160ms ease-out',
        beat: 'beatDecay 500ms linear forwards',
      },
    },
  },
  plugins: [],
} satisfies Config;
