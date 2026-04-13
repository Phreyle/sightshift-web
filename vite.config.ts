import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    // Prevent Vite from pre-bundling the WASM-containing mediapipe package
    exclude: ['@mediapipe/tasks-vision'],
  },
})
