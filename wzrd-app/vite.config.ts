import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Tauri's expected port matches `tauri.conf.json`'s `devUrl`.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: false,
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: 'esnext',
    minify: false,
    sourcemap: true,
  },
});
