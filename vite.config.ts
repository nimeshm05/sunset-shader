import { defineConfig } from "vite";

export default defineConfig({
  // Pure WebGL2 project: no plugins required.
  // GLSL sources are imported as raw strings via the `?raw` suffix.
  server: {
    open: false,
  },
});
