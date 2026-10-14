/**
 * The footer and notifications, by pi's mode, following the daemon's real
 * events.
 */

import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, vi } from "vitest";

import { SPINNER_FRAMES, STATIC_FRAME } from "../extensions/rbtr/footer.js";
import { HEALTH_CHECK_MS } from "../extensions/rbtr/index.js";
import { test, until } from "./session.fixtures.js";

/** The footer of an index with no embeddings, its daemon running or gone; "23.9k symbols" or "812 symbols". */
const INDEXED = /^rbtr: ○ [\d.]+k? symbols · not embedded$/;
const INDEXED_NO_DAEMON = /^rbtr: ○ [\d.]+k? symbols · no daemon · not embedded$/;

test("in the terminal UI, a footer that waits on rbtr animates", async ({ heldRbtr: _, pi }) => {
  await pi.start({ mode: "tui" });
  await new Promise((resolve) => setTimeout(resolve, 1000));
  const starting = pi.ui.statuses.filter((status) => status.includes("starting"));
  expect(new Set(starting).size).toBeGreaterThan(5);
  expect(starting.every((status) => SPINNER_FRAMES.some((frame) => status === `rbtr: ${frame} starting…`))).toBe(true);
});

// In RPC every status is a message on stdout: one per state.
test("in rpc mode, a footer that waits on rbtr shows one static status", async ({ heldRbtr: _, pi }) => {
  await pi.start({ mode: "rpc" });
  await new Promise((resolve) => setTimeout(resolve, 1000));
  const starting = pi.ui.statuses.filter((status) => status.includes("starting"));
  expect(starting).toEqual([`rbtr: ${STATIC_FRAME} starting…`]);
});

describe("in a repository of the test's own", () => {
  test.override({ projectSettings: { autoIndex: false }, longBuild: true });
  test.override("cwd", ({ writableRepo }) => writableRepo);

  test("a build's progress and its end reach the footer", async ({ pi }) => {
    await pi.start();
    const before = pi.ui.statuses.length;
    await pi.run({ name: "rbtr_watch", args: { refs: ["HEAD"] } });
    // The build's end: the label after the "indexing" status.
    await until(() => {
      const since = pi.ui.statuses.slice(before);
      const building = since.findIndex((status) => status.includes("indexing"));
      return building !== -1 && since.slice(building + 1).some((status) => INDEXED.test(status));
    });
  });
});

describe("with a daemon of its own", () => {
  test.override({ ownDaemon: true });

  // The health check runs every `HEALTH_CHECK_MS`; only its interval is faked, so it can
  // be run at once while every process stays real.
  test("the footer follows the daemon stopping and coming back", async ({ cli, pi }) => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    try {
      const last = () => pi.ui.statuses.at(-1) ?? "";
      await pi.start();
      await until(() => INDEXED.test(last()));
      cli("daemon", "stop");
      vi.advanceTimersByTime(HEALTH_CHECK_MS);
      await until(() => INDEXED_NO_DAEMON.test(last()));
      cli("daemon", "start");
      vi.advanceTimersByTime(HEALTH_CHECK_MS);
      await until(() => INDEXED.test(last()));
    } finally {
      vi.useRealTimers();
    }
  });

  test("a health check that ends after its session was replaced reports nothing", async ({
    heldRbtr,
    processes,
    temp,
    cli,
    unhandledRejections,
    pi,
  }) => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    try {
      const last = () => pi.ui.statuses.at(-1) ?? "";
      const probes = () => processes().filter((line) => line === "held-rbtr --json daemon status").length;
      heldRbtr.release();
      await pi.start();
      await until(() => INDEXED.test(last()));
      cli("daemon", "stop");
      vi.advanceTimersByTime(HEALTH_CHECK_MS);
      await until(() => / · no daemon/.test(last()));
      cli("daemon", "start");
      // `held-rbtr` reads its release file when it starts: the next health
      // check's look for the daemon waits for this one.
      const release = join(temp, "release-again");
      process.env.RBTR_RELEASE = release;
      const before = probes();
      vi.advanceTimersByTime(HEALTH_CHECK_MS);
      await until(() => probes() > before);
      await pi.runtime.newSession();
      writeFileSync(release, "");
      // Long enough for the health check to end.
      await new Promise((resolve) => setTimeout(resolve, 1000));
      expect(unhandledRejections).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });
});

// A real daemon publishes an index error only when a commit vanishes between
// resolving a ref and building it, so this one runs on `FakeDaemon`.
test("an index error is notified, and the footer says so", async ({ repo, fakeDaemon, pi }) => {
  const message = "Cannot resolve ref 'gone'";
  await pi.start();
  await until(() => / symbols$/.test(pi.ui.statuses.at(-1) ?? ""));
  // Published until it arrives: a subscriber misses what is sent before it joins.
  await until(() => {
    void fakeDaemon.publish({ kind: "index_error", repo_path: repo, message });
    return pi.ui.notifications.includes(message);
  });
  expect(pi.ui.statuses.at(-1)).toBe("rbtr: ✗ error");
  // pi's own footer shows it.
  expect(pi.footer(120).at(-1)).toContain("rbtr: ✗ error");
});

// The daemon publishes gc's notifications to every subscriber, whichever repos it collects.
test("a gc's progress reaches the footer, and its end restores the index label", async ({ fakeDaemon, pi }) => {
  await pi.start();
  await until(() => / symbols$/.test(pi.ui.statuses.at(-1) ?? ""));
  const label = pi.ui.statuses.at(-1);
  // Published until it arrives: a subscriber misses what is sent before it joins.
  await until(() => {
    void fakeDaemon.publish({ kind: "gc_progress", current: 1, total: 2 });
    return / collecting 1\/2…$/.test(pi.ui.statuses.at(-1) ?? "");
  });
  await fakeDaemon.publish({ kind: "gc_ended" });
  await until(() => pi.ui.statuses.at(-1) === label);
});

describe("in rpc mode", () => {
  test("/rbtr-settings says it needs the terminal UI, and opens nothing", async ({ pi }) => {
    await pi.start({ mode: "rpc" });
    // The recording UI's `custom` throws: the settings screen must not open.
    await pi.runtime.session.prompt("/rbtr-settings");
    expect(pi.ui.notifications).toEqual(["/rbtr-settings requires the interactive terminal UI"]);
  });
});
