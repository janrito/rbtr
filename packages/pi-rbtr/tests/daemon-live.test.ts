/**
 * Integration test — spawns a real ``rbtr daemon`` subprocess
 * and drives it through the production TypeScript client.
 *
 * Widest sensible entry point: no fakes on either side.  Uses a
 * disposable ``RBTR_HOME`` tmpdir so nothing touches the user's
 * real `~/.rbtr`.
 */

import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, expect, test } from "vitest";

import { queryDaemonStatus, send } from "../extensions/rbtr/daemon-client.js";
import { resolveCommand } from "../extensions/rbtr/exec.js";

// macOS socket path limit — keep prefix short.
const home = mkdtempSync(join(tmpdir(), `rbtr-${randomBytes(3).toString("hex")}-`));
const repo = mkdtempSync(join(tmpdir(), `rbtr-r-${randomBytes(3).toString("hex")}-`));

const RBTR = resolveCommand("rbtr");
const env = { ...process.env, RBTR_HOME: home, RBTR_WARMUP: "false" };

beforeAll(() => {
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

  const start = spawnSync(RBTR.executable, [...RBTR.baseArgs, "daemon", "start"], { env, encoding: "utf8" });
  if (start.status !== 0) {
    throw new Error(`daemon start failed (code=${start.status}):\nstdout: ${start.stdout}\nstderr: ${start.stderr}`);
  }
});

afterAll(() => {
  spawnSync(RBTR.executable, [...RBTR.baseArgs, "daemon", "stop"], { env, encoding: "utf8" });
  rmSync(home, { recursive: true, force: true });
  rmSync(repo, { recursive: true, force: true });
});

test("queryDaemonStatus reports running with pid + rpc endpoint", async () => {
  // queryDaemonStatus always shells out to the CLI; inherit the
  // test's RBTR_HOME so it sees the same daemon we started.
  const previous = process.env.RBTR_HOME;
  process.env.RBTR_HOME = home;
  try {
    const status = await queryDaemonStatus();
    expect(status.running).toBe(true);
    expect(status.pid).toBeGreaterThan(0);
    expect(status.rpc).toMatch(/^ipc:\/\//);
  } finally {
    process.env.RBTR_HOME = previous;
  }
});

test("send(StatusRequest) round-trips against a real daemon", async () => {
  process.env.RBTR_HOME = home;
  try {
    const response = await send({ kind: "status", path: repo });
    expect(response.kind).toBe("status");
    if (response.kind === "status") {
      // Fresh daemon, freshly-initialised repo → no index yet.
      expect(response.indexed_refs).toEqual([]);
    }
  } finally {
    process.env.RBTR_HOME = home;
  }
});
