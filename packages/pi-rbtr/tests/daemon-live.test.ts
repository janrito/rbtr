/**
 * Integration test — spawns a real ``rbtr daemon`` subprocess
 * and drives it through the production TypeScript client.
 *
 * Widest sensible entry point: no fakes on either side.  The
 * daemon is isolated via the ``RBTR_DATA_DIR`` / ``RBTR_CONFIG_DIR``
 * / ``RBTR_LOG_DIR`` overrides so it gets its own index DB, runtime
 * sockets, and logs — nothing touches the user's real daemon.  The
 * model cache is left shared so weights are not re-downloaded, and
 * ``RBTR_WARMUP=false`` stops the test daemon loading any models.
 */

import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, expect, test } from "vitest";

import { queryDaemonStatus, send } from "../extensions/rbtr/daemon-client.js";
import { resolveCommand } from "../extensions/rbtr/exec.js";

const dataDir = mkdtempSync(join(tmpdir(), `rbtr-d-${randomBytes(3).toString("hex")}-`));
const configDir = mkdtempSync(join(tmpdir(), `rbtr-c-${randomBytes(3).toString("hex")}-`));
const logDir = mkdtempSync(join(tmpdir(), `rbtr-l-${randomBytes(3).toString("hex")}-`));
const repo = mkdtempSync(join(tmpdir(), `rbtr-r-${randomBytes(3).toString("hex")}-`));

const RBTR = resolveCommand("rbtr");

// runtime_dir (and thus the sockets) derives from data_dir, so
// redirecting RBTR_DATA_DIR isolates both the index DB and the RPC
// endpoint.  Sockets still live under platformdirs' user_runtime_path,
// so the macOS socket-path length limit is not a concern here.
const overrides: Record<string, string> = {
  RBTR_DATA_DIR: dataDir,
  RBTR_CONFIG_DIR: configDir,
  RBTR_LOG_DIR: logDir,
  RBTR_WARMUP: "false",
};
const savedEnv: Record<string, string | undefined> = {};

beforeAll(() => {
  // queryDaemonStatus / send shell out to the CLI inheriting
  // process.env, so the overrides have to live there for the whole
  // file (restored in afterAll).  Safe: the file runs in its own
  // forked worker.
  for (const [key, value] of Object.entries(overrides)) {
    savedEnv[key] = process.env[key];
    process.env[key] = value;
  }

  // handle_status on the daemon calls git.open_repo, so the test
  // repo has to be a real git directory with at least one commit.
  const g = (args: string[]) => {
    const r = spawnSync("git", args, { cwd: repo, encoding: "utf8" });
    if (r.status !== 0) throw new Error(`git ${args.join(" ")}: ${r.stderr}`);
  };
  g(["init", "-q", "-b", "main"]);
  g(["config", "user.email", "t@t.t"]);
  g(["config", "user.name", "t"]);
  g(["commit", "--allow-empty", "-qm", "init"]);

  const start = spawnSync(RBTR.executable, [...RBTR.baseArgs, "daemon", "start"], { encoding: "utf8" });
  if (start.status !== 0) {
    throw new Error(`daemon start failed (code=${start.status}):\nstdout: ${start.stdout}\nstderr: ${start.stderr}`);
  }
});

afterAll(() => {
  spawnSync(RBTR.executable, [...RBTR.baseArgs, "daemon", "stop"], { encoding: "utf8" });
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  rmSync(dataDir, { recursive: true, force: true });
  rmSync(configDir, { recursive: true, force: true });
  rmSync(logDir, { recursive: true, force: true });
  rmSync(repo, { recursive: true, force: true });
});

test("queryDaemonStatus reports running with pid + rpc endpoint", async () => {
  const status = await queryDaemonStatus();
  expect(status.running).toBe(true);
  expect(status.pid).toBeGreaterThan(0);
  expect(status.rpc).toMatch(/^ipc:\/\//);
});

test("send(StatusRequest) round-trips against a real daemon", async () => {
  const response = await send({ kind: "status", repo_path: repo });
  expect(response.kind).toBe("status");
  if (response.kind === "status") {
    // Isolated daemon, freshly-initialised repo → no index yet.
    expect(response.indexed_refs).toEqual([]);
  }
});
