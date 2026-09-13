import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// JLA_UI_PORT / JLA_API_TARGET let a test instance run beside the everyday one.
const port = Number(process.env.JLA_UI_PORT || 5173);
const target = process.env.JLA_API_TARGET || "http://127.0.0.1:8077";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port,
    strictPort: true,
    proxy: { "/api": { target, changeOrigin: false } },
  },
  preview: { host: "127.0.0.1", port, proxy: { "/api": { target, changeOrigin: false } } },
});
