import { defineConfig, devices } from "@playwright/test";

// Local iteration uses the fast dev server; CI builds once and serves the
// production bundle (GitHub Actions sets CI=true automatically).
const webServerCommand = process.env.CI
  ? "npm run build && npm run start -- -p 3100"
  : "npm run dev -- -p 3100";

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["html"], ["list"]] : [["html"], ["list"]],
  use: {
    baseURL: "http://localhost:3100",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: webServerCommand,
    port: 3100,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
