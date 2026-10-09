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
      keyframes: {
        pulseArm: {
          '0%,100%': { boxShadow: '0 0 0 1px rgb(var(--live) / .7), 0 0 18px -4px rgb(var(--live) / .6)' },
          '50%': { boxShadow: '0 0 0 1px rgb(var(--live) / 1), 0 0 30px -2px rgb(var(--live) / .95)' },
        },
        fadeIn: { from: { opacity: '0', transform: 'translateY(4px)' }, to: { opacity: '1', transform: 'none' } },
      },
      animation: {
        arm: 'pulseArm 1.6s ease-in-out infinite',
        in: 'fadeIn 160ms ease-out',
      },
    },
  },
  plugins: [],
} satisfies Config;
