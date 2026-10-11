/**
 * How the extension talks to its daemon: when the daemon does not answer
 * (a real daemon, stopped with SIGSTOP, is alive and silent), and when it
 * is subscribed to twice.
 */

import { readFileSync, realpathSync } from "node:fs";
import { join } from "node:path";
import { expect, inject } from "vitest";

import { send } from "../extensions/rbtr/daemon-client.js";
import { DaemonSession } from "../extensions/rbtr/daemon-session.js";
import type { Notification } from "../extensions/rbtr/generated/protocol.js";
import { test, until } from "./session.fixtures.js";

// Every test here needs a daemon started for it, and most freeze it.
test.override({ ownDaemon: true });

/** How many status requests for *repo* the daemon logged as served; it logs the real path. */
function statusRequestsServed(logDir: string, repo: string): number {
  const root = realpathSync(repo);
  return readFileSync(join(logDir, "daemon.log"), "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line) as { event: string; kind?: string; repo?: string })
    .filter((entry) => entry.event === "request_complete" && entry.kind === "status" && entry.repo === root).length;
}

test("a late reply is waited for, and the daemon is asked once", async ({
  rbtr,
  repo,
  runningDaemon,
  frozenDaemon,
}) => {
  const before = statusRequestsServed(rbtr.logDir, repo);
  frozenDaemon.freeze();
  const reply = send(
    { kind: "status", repo_path: repo },
    { rpcEndpoint: runningDaemon.rpc ?? "", waitBudgetMs: 10_000 },
  );
  await new Promise((resolve) => setTimeout(resolve, 500));
  frozenDaemon.thaw();
  expect(await reply).toMatchObject({ kind: "status" });
  expect(statusRequestsServed(rbtr.logDir, repo) - before).toBe(1);
});

test("waiting stops when the budget is spent", async ({ repo, runningDaemon, frozenDaemon }) => {
  frozenDaemon.freeze();
  const started = Date.now();
  await expect(
    send({ kind: "status", repo_path: repo }, { rpcEndpoint: runningDaemon.rpc ?? "", waitBudgetMs: 300 }),
  ).rejects.toMatchObject({ code: "EAGAIN" });
  expect(Date.now() - started).toBeLessThan(2_000);
});

test("an abort ends the wait at once", async ({ repo, runningDaemon, frozenDaemon }) => {
  frozenDaemon.freeze();
  const controller = new AbortController();
  const started = Date.now();
  const reply = send(
    { kind: "status", repo_path: repo },
    { rpcEndpoint: runningDaemon.rpc ?? "", signal: controller.signal },
  );
  setTimeout(() => controller.abort(), 100);
  await expect(reply).rejects.toMatchObject({ name: "AbortError" });
  expect(Date.now() - started).toBeLessThan(1_000);
});

test("a tool call aborted while the daemon is silent ends at once, and nothing else is tried", async ({
  piReady: pi,
  frozenDaemon,
  processes,
}) => {
  const before = processes().length;
  frozenDaemon.freeze();
  let running = false;
  const unsubscribe = pi.runtime.session.subscribe((event) => {
    if (event.type === "tool_execution_start") running = true;
  });
  const waiting = pi.run({ name: "rbtr_read_symbol", args: { symbol: inject("corpus").subject.name } });
  await until(() => running);
  unsubscribe();
  const started = Date.now();
  await pi.runtime.session.abort();
  const [aborted] = await waiting;
  expect(Date.now() - started).toBeLessThan(1_000);
  expect(aborted?.isError).toBe(true);
  // Neither a CLI fallback nor a look for another daemon.
  expect(processes().slice(before)).toEqual([]);
});

test("a second subscription to the daemon adds no second handler", async ({ rbtr, repo, runningDaemon }) => {
  const session = new DaemonSession();
  session.useCommand({ executable: "rbtr", baseArgs: ["--json"], description: "rbtr" }, runningDaemon);
  const first: Notification[] = [];
  const second: Notification[] = [];
  session.subscribe((notification) => first.push(notification));
  session.subscribe((notification) => second.push(notification));
  try {
    // ZMQ's subscriber needs a moment to join before anything is published.
    await new Promise((resolve) => setTimeout(resolve, 200));
    // A ref that needs a build: the daemon publishes its progress.
    rbtr.cli("watch", "--repo-path", repo, "unindexed");
    await until(() => first.length > 0);
    expect(second).toEqual([]);
  } finally {
    session.stopSubscribing();
    session.stopSubscribing();
  }
});
