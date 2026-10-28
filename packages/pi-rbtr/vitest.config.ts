import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { defineConfig, type Plugin } from "vitest/config";

/**
 * The package directory and version of the installed `pi`, as pi itself
 * reports them to an extension (`tests/fixtures/report-pi.ts`). pi loads it
 * and stops at the model it cannot find, with an empty agent directory so
 * nothing of the user's is read. The installed `pi` is the one on PATH
 * outside `node_modules/.bin`, which `npm exec` puts first and which holds
 * the workspace's own pi, the one `tsc` checks against.
 */
function installedPi(): { dir: string; version: string } {
  const temp = mkdtempSync(join(tmpdir(), "pi-report-"));
  const report = join(temp, "report.json");
  const PATH = (process.env.PATH ?? "")
    .split(delimiter)
    .filter((dir) => !dir.endsWith(join("node_modules", ".bin")))
    .join(delimiter);
  try {
    execFileSync(
      "pi",
      ["-ne", "-e", "tests/fixtures/report-pi.ts", "-p", "--no-session", "--model", "none/none", "report"],
      { env: { ...process.env, PATH, PI_CODING_AGENT_DIR: temp, PI_REPORT: report }, stdio: "ignore" },
    );
  } catch {
    // pi exits 1 at the missing model, after loading the extension.
  }
  try {
    return JSON.parse(readFileSync(report, "utf8"));
  } catch {
    throw new Error("pi-rbtr's tests run against the installed pi, and `pi` reported none: install pi");
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
}

const pi = installedPi();

/**
 * Resolve pi's packages and `typebox` as the installed pi resolves them, so
 * the extension and the tests run on the pi that loads the extension.
 */
const fromInstalledPi: Plugin = {
  name: "installed-pi",
  enforce: "pre",
  resolveId(source) {
    if (!/^(?:@earendil-works\/|typebox(?:\/|$))/.test(source)) return null;
    return this.resolve(source, join(pi.dir, "package.json"), { skipSelf: true });
  },
};

export default defineConfig({
  plugins: [fromInstalledPi],
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // The run's repository, its index, and the run's shared daemon.
    globalSetup: ["tests/global-setup.ts"],
    provide: { piVersion: pi.version },
    fileParallelism: true,
    // A transformed module keeps the pi it was resolved to, and its cache key
    // is its own content, so a module unchanged across a pi upgrade would go
    // on loading the old pi. A cache per pi version starts afresh instead.
    fsModuleCachePath: join("node_modules", ".vitest-cache", `pi-${pi.version}`),
    // A test with a daemon of its own (`ownDaemon`) indexes a copy of the
    // repository and starts that daemon: 7–10 s at load average 24, and over
    // 20 s at load average 15 with the Python suite running. The timeout only
    // bounds a hang, so it leaves room for both.
    testTimeout: 90_000,
  },
});
