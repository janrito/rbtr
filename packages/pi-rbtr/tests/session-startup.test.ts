/**
 * A pi session's lifecycle: finding rbtr and its daemon in the background,
 * the project settings it trusts, what it adds to the prompt, the tools it
 * starts with.
 */

import { existsSync } from "node:fs";
import { join } from "node:path";
import type { SystemMessage } from "@earendil-works/pi-ai";
import { describe, expect, inject } from "vitest";

import { HEADER } from "../extensions/rbtr/annotate.js";
import { ON_REQUEST } from "../extensions/rbtr/tool-set.js";
import { type Call, DAEMON_AT, test, textOf, until } from "./session.fixtures.js";

/** Whether *promise* settles within *ms*. */
async function settlesWithin(promise: Promise<unknown>, ms: number): Promise<boolean> {
  const timeout = new Promise<false>((resolve) => setTimeout(() => resolve(false), ms));
  const settled = promise.then(
    () => true,
    () => true,
  );
  return Promise.race([settled, timeout]);
}

const { subject } = inject("corpus");

const readSubject: Call = { name: "rbtr_read_symbol", args: { symbol: subject.name } };

// A stopped daemon is the test's own: the shared one keeps running.
describe.for<{
  daemon: "running" | "stopped";
  missing: ("python3" | "rbtr" | "uvx")[];
  started: string[];
}>([
  { daemon: "running", missing: [], started: ["python3 -m rbtr --json daemon status"] },
  {
    daemon: "stopped",
    missing: [],
    started: ["python3 -m rbtr --json daemon status", "python3 -m rbtr --json daemon start"],
  },
  {
    daemon: "running",
    missing: ["python3"],
    started: ["python3 -m rbtr --json daemon status", "rbtr --json daemon status"],
  },
])("with the daemon $daemon and $missing missing", ({ daemon, missing, started }) => {
  test.override({ ownDaemon: daemon === "stopped", missing });

  test("startup runs the fewest rbtr processes, and the daemon serves the tools", async ({ processes, pi }) => {
    await pi.start();
    const [end] = await pi.run(readSubject);
    expect(end.isError).toBe(false);
    expect(processes()).toEqual(started);
  });
});

// Startup indexes a repository it finds unindexed, or cannot tell about
// because rbtr cannot read its index, only when autoIndex is on.
describe.for<{ index: "unindexed" | "unreadable"; settings: { autoIndex: boolean; command?: string }; footer: RegExp }>(
  [
    { index: "unindexed", settings: { autoIndex: true }, footer: / symbols/ },
    { index: "unindexed", settings: { autoIndex: false }, footer: /^rbtr: no index — \/rbtr-index$/ },
    {
      index: "unreadable",
      settings: { autoIndex: false, command: DAEMON_AT },
      footer: /^rbtr: index unavailable — retrying$/,
    },
  ],
)("in a repository whose index is $index, with autoIndex $settings.autoIndex", ({ settings, footer }) => {
  test.override({ projectSettings: settings });
  test.override("cwd", ({ writableRepo }) => writableRepo);

  test("startup indexes only when autoIndex is on", async ({ pi }) => {
    await pi.start();
    await until(() => footer.test(pi.ui.statuses.at(-1) ?? ""));
  });
});

describe("with no rbtr installed", () => {
  test.override({ missing: ["python3", "rbtr", "uvx"] });

  test("startup tries the active Python, then PATH, then uvx, then suggests installing rbtr", async ({
    processes,
    pi,
  }) => {
    await pi.start();
    const [end] = await pi.run(readSubject);
    expect(end.isError).toBe(true);
    // Nothing is started or indexed once no candidate runs.
    expect(processes()).toEqual([
      "python3 -m rbtr --json daemon status",
      "rbtr --json daemon status",
      "uvx rbtr --json daemon status",
    ]);
    expect(pi.ui.statuses.at(-1)).toBe("rbtr: cannot run");
    expect(pi.ui.notifications.join("\n")).toMatch(
      /rbtr could not be run \(uvx rbtr: .+\)\. It may not be installed: uv tool install rbtr/,
    );
  });
});

test("startup runs in the background: a tool waits for it, and an abort ends the wait", async ({ heldRbtr, pi }) => {
  // Returns although rbtr has not answered.
  await pi.start();
  const waiting = pi.run(readSubject);
  expect(await settlesWithin(waiting, 500)).toBe(false);
  await pi.runtime.session.abort();
  const [aborted] = await waiting;
  expect(aborted?.isError).toBe(true);
  heldRbtr.release();
  const [answered] = await pi.run(readSubject);
  expect(answered.isError).toBe(false);
  expect(answered.result.structuredContent).toMatchObject({ kind: "read_symbol" });
});

test("a bash result that arrives during startup gets its note once startup ends", async ({ heldRbtr, pi }) => {
  await pi.start();
  const running = pi.run({ name: "bash", args: { command: `grep -rln ${subject.name} ${subject.user}` } });
  expect(await settlesWithin(running, 1000)).toBe(false);
  heldRbtr.release();
  const [end] = await running;
  expect(textOf(end.result)).toContain(HEADER);
});

describe.for([
  { projectTrusted: true, first: "held-rbtr --json daemon status" },
  { projectTrusted: false, first: "python3 -m rbtr --json daemon status" },
] as const)("in a project trusted: $projectTrusted", ({ projectTrusted, first }) => {
  test.override({ projectTrusted, settingsScope: "project" });
  test.override("cwd", ({ writableRepo }) => writableRepo);

  test("the project's rbtr settings apply only when trusted", async ({ heldRbtr, processes, pi }) => {
    heldRbtr.release();
    await pi.start();
    await until(() => processes().length > 0);
    expect(processes()[0]).toBe(first);
  });
});

describe("in an untrusted project", () => {
  test.override({ projectTrusted: false });
  test.override("cwd", ({ writableRepo }) => writableRepo);

  test("/rbtr-settings edits nothing, and says why", async ({ writableRepo, pi }) => {
    await pi.start();
    // The recording UI's `custom` throws: the settings dialog must not open.
    await pi.runtime.session.prompt("/rbtr-settings");
    expect(pi.ui.notifications.join("\n")).toMatch(/trusted/);
    expect(existsSync(join(writableRepo, ".pi", "rbtr-index.json"))).toBe(false);
  });
});

test("the index line is pi's rbtr prompt section, and replaces no prompt", async ({ pi }) => {
  await pi.start();
  await pi.run();
  const system = pi.runtime.session.messages.find((message): message is SystemMessage => message.role === "system");
  expect(system?.sections?.rbtr).toMatch(/rbtr code index/);
  expect(system?.content).toBe("");
});

test("the housekeeping tools start inactive, and rbtr_index_tools makes them active", async ({ pi }) => {
  await pi.start();
  const active = () => pi.runtime.session.getActiveToolNames();
  expect(active()).toEqual(
    expect.arrayContaining([
      "rbtr_search",
      "rbtr_read_symbol",
      "rbtr_find_refs",
      "rbtr_changed_symbols",
      "rbtr_list_symbols",
      "rbtr_watch",
      "rbtr_index_tools",
    ]),
  );
  expect(active().filter((name) => ON_REQUEST.includes(name))).toEqual([]);
  await pi.run({ name: "rbtr_index_tools", args: {} });
  expect(active()).toEqual(expect.arrayContaining([...ON_REQUEST]));
});

describe("with an rbtr that cannot read its index, and its index held", () => {
  test.override({ heldProgram: DAEMON_AT, holdOn: "watch" });

  // Startup cannot tell whether the repository is indexed, so it indexes;
  // that fails, after the session was replaced.
  test("an index that fails after its session was replaced reports nothing", async ({
    heldRbtr,
    processes,
    unhandledRejections,
    pi,
  }) => {
    await pi.start();
    await until(() => processes().some((line) => line.startsWith("held-rbtr --json watch")));
    await pi.runtime.newSession();
    heldRbtr.release();
    await until(() => processes().some((line) => line.startsWith("daemon-at --json watch")));
    // Long enough for the failure to reach the extension.
    await new Promise((resolve) => setTimeout(resolve, 500));
    expect(unhandledRejections).toEqual([]);
  });
});

// pi 1.0 does not restore the tools a session loaded when it builds another
// from it (a fork, a resume): `_restoreToolsFromTranscript` runs only for a
// session given no starting tools, and the SDK always gives them. The model
// loads them again. If this fails, pi restores them now, and this test, the
// ARCHITECTURE note and the loader's guideline should say so.
test("a fork starts without the loaded housekeeping tools, and the loader loads them again", async ({ pi }) => {
  await pi.start();
  await pi.run({ name: "rbtr_index_tools", args: {} });
  const leaf = pi.runtime.session.sessionManager.getLeafId();
  if (leaf === null) throw new Error("the session has no entries");
  await pi.runtime.fork(leaf, { position: "at" });
  await pi.start();
  const active = () => pi.runtime.session.getActiveToolNames();
  expect(active().filter((name) => ON_REQUEST.includes(name))).toEqual([]);
  await pi.run({ name: "rbtr_index_tools", args: {} });
  expect(active()).toEqual(expect.arrayContaining([...ON_REQUEST]));
});
