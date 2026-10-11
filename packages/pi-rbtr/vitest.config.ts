import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // The run's repository, its index, and the run's shared daemon.
    globalSetup: ["tests/global-setup.ts"],
    fileParallelism: true,
    // A test with a daemon of its own (`ownDaemon`) indexes a copy of the
    // repository and starts that daemon: 7–10 s at load average 24.
    testTimeout: 30_000,
  },
});
