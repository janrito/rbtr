/**
 * Tests for {@link DaemonSession.reconcile} and
 * {@link compareVersions}.
 *
 * `reconcile` decides by version, and a real daemon cannot be made to
 * report another version than the extension's; so the start and stop
 * hooks and `daemon status` are scripted here. A real start, and a daemon
 * already up to date, are tested through sessions in
 * `session-startup.test.ts`.
 */

import { test as base, describe, expect } from "vitest";

import type { DaemonStatus } from "../extensions/rbtr/daemon-client.js";
import {
  compareVersions,
  DaemonSession,
  type ReconcileDeps,
  type ReconcileResult,
} from "../extensions/rbtr/daemon-session.js";
import type { ResolvedCommand } from "../extensions/rbtr/exec.js";

describe("compareVersions", () => {
  test.each([
    ["2026.4.0", "2026.4.0", 0],
    ["2026.4.1", "2026.4.0", 1],
    ["2026.4.0", "2026.4.1", -1],
    ["2026.5.0", "2026.4.9", 1],
    ["2025.12.9", "2026.1.0", -1],
    [null, "2026.4.0", -1],
    ["2026.4.0", null, 1],
    [null, null, 0],
    ["2026.10.0", "2026.9.3", 1],
    ["2026.9.0", "2026.9.0.dev1", 1],
    ["2026.9.0-dev1", "2026.9.0.dev1", 0],
    ["2026.9.0.dev2", "2026.9.0-dev1", 1],
    ["not a version", "2026.9.0", null],
  ] as const)("compareVersions(%o, %o) -> %o", (a, b, expected) => {
    const result = compareVersions(a, b);
    expect(result === null ? null : Math.sign(result)).toBe(expected);
  });
});

const test = base.extend<{
  found: string | null;
  refreshed: (string | null)[];
  printed: string | null;
  fails: "start" | "stop" | null;
  daemon: Omit<DaemonStatus, "version">;
  command: ResolvedCommand;
  session: DaemonSession;
  hooks: { deps: ReconcileDeps; calls: ("start" | "stop")[] };
}>({
  // The version of the daemon the session found at startup; none when null.
  found: null,
  // The versions `daemon status` reports afterwards, in order; none when null.
  refreshed: [],
  // The version of the daemon `daemon start` prints; none when null.
  printed: null,
  // The hook that fails.
  fails: null,

  // A running daemon, but for its version.
  daemon: { running: true, pid: 1, rpc: "ipc:///tmp/test.rpc", pub: "ipc:///tmp/test.pub" },

  command: { executable: "rbtr", baseArgs: ["--json"], description: "rbtr" },

  session: async ({ found, refreshed, daemon, command }, use) => {
    const session = new DaemonSession();
    session.useCommand(command, found === null ? null : { ...daemon, version: found });
    let step = 0;
    session.refresh = async () => {
      const version = refreshed[step++] ?? null;
      const status = version === null ? null : { ...daemon, version };
      session.useCommand(command, status);
      return status;
    };
    await use(session);
  },

  // The start and stop hooks, and the calls made to them.
  hooks: async ({ printed, fails, daemon }, use) => {
    const calls: ("start" | "stop")[] = [];
    await use({
      calls,
      deps: {
        startDaemon: async () => {
          calls.push("start");
          if (fails === "start") throw new Error("simulated start failure");
          return printed === null ? null : { ...daemon, version: printed };
        },
        stopDaemon: async () => {
          calls.push("stop");
          if (fails === "stop") throw new Error("simulated stop failure");
        },
      },
    });
  },
});

// The daemon found at startup, what `daemon status` reports afterwards, what
// `daemon start` prints, which hook fails, and the extension's version.
describe.for<
  [
    string,
    {
      found: string | null;
      refreshed: (string | null)[];
      printed: string | null;
      fails: "start" | "stop" | null;
      extension: string;
      calls: ("start" | "stop")[];
      result: Partial<ReconcileResult>;
    },
  ]
>([
  [
    "not running, start prints no status: started, as daemon status then reports",
    {
      found: null,
      refreshed: ["2026.4.0"],
      printed: null,
      fails: null,
      extension: "2026.4.0",
      calls: ["start"],
      result: { outcome: "started", newVersion: "2026.4.0" },
    },
  ],
  [
    "not running, start fails: failed, with the reason",
    {
      found: null,
      refreshed: [],
      printed: null,
      fails: "start",
      extension: "2026.4.0",
      calls: ["start"],
      result: { outcome: "failed", detail: expect.stringContaining("simulated start failure") },
    },
  ],
  [
    "extension older than the daemon: left running",
    {
      found: "2026.5.0",
      refreshed: [],
      printed: null,
      fails: null,
      extension: "2026.4.0",
      calls: [],
      result: { outcome: "older_client", newVersion: "2026.5.0" },
    },
  ],
  [
    "extension newer than the daemon: restarted",
    {
      found: "2026.4.0",
      refreshed: [],
      printed: "2026.5.0",
      fails: null,
      extension: "2026.5.0",
      calls: ["stop", "start"],
      result: { outcome: "restarted", previousVersion: "2026.4.0", newVersion: "2026.5.0" },
    },
  ],
  [
    "extension newer by a month past September: restarted",
    {
      found: "2026.9.0.dev1",
      refreshed: [],
      printed: "2026.10.0",
      fails: null,
      extension: "2026.10.0",
      calls: ["stop", "start"],
      result: { outcome: "restarted", newVersion: "2026.10.0" },
    },
  ],
  [
    "extension older by a month past September: left running",
    {
      found: "2026.10.0",
      refreshed: [],
      printed: null,
      fails: null,
      extension: "2026.9.0-dev1",
      calls: [],
      result: { outcome: "older_client" },
    },
  ],
  [
    "a version that cannot be read: left running",
    {
      found: "local build",
      refreshed: [],
      printed: null,
      fails: null,
      extension: "2026.9.0",
      calls: [],
      result: { outcome: "up_to_date" },
    },
  ],
  [
    "extension newer, stop fails: failed",
    {
      found: "2026.4.0",
      refreshed: [],
      printed: null,
      fails: "stop",
      extension: "2026.5.0",
      calls: ["stop"],
      result: { outcome: "failed" },
    },
  ],
  [
    "extension newer, start prints no status and daemon status finds none: failed",
    {
      found: "2026.4.0",
      refreshed: [null],
      printed: null,
      fails: null,
      extension: "2026.5.0",
      calls: ["stop", "start"],
      result: { outcome: "failed" },
    },
  ],
])("DaemonSession.reconcile, %s", ([, row]) => {
  test.override({ found: row.found, refreshed: row.refreshed, printed: row.printed, fails: row.fails });

  test("ends as the row says, calling the hooks it names", async ({ session, hooks }) => {
    expect(await session.reconcile(row.extension, hooks.deps)).toMatchObject(row.result);
    expect(hooks.calls).toEqual(row.calls);
  });
});
