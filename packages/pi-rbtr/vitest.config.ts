import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		environment: "node",
		include: ["tests/**/*.test.ts"],
		// One worker at a time for tests that spawn subprocess daemons;
		// most tests opt back into concurrency via `describe.concurrent`.
		fileParallelism: true,
		testTimeout: 15_000,
	},
});
