import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Engine-room palette — dark, terminal-ish. Phase 4.1 doesn't try to
        // be pretty; it tries to be legible at 2m from a projector laptop.
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
          green: '#7ddf86',
          amber: '#f0c065',
          red: '#ff7568',
          blue: '#73b6ff',
          violet: '#b78cff',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config;
