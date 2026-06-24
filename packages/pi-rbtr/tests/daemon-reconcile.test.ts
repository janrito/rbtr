/**
 * Tests for {@link DaemonSession.reconcile} and
 * {@link compareCalver}.
 *
 * `reconcile` is pure state-transition logic around a daemon
 * lifecycle; we mock the `execDaemon` hook and drive the
 * session's observed status through a small scriptable object,
 * so no real daemon or CLI is involved.
 */

import { describe, expect, test } from "vitest";

import { compareCalver, DaemonSession, type ReconcileDeps } from "../extensions/rbtr/daemon-session.js";

describe("compareCalver", () => {
  test.each([
    ["2026.4.0", "2026.4.0", 0],
    ["2026.4.1", "2026.4.0", 1],
    ["2026.4.0", "2026.4.1", -1],
    ["2026.5.0", "2026.4.9", 1],
    ["2025.12.9", "2026.1.0", -1],
    [null, "2026.4.0", -1],
    ["2026.4.0", null, 1],
    [null, null, 0],
  ] as const)("compareCalver(%o, %o) -> %i", (a, b, expected) => {
    const result = compareCalver(a, b);
    if (expected === 0) expect(result).toBe(0);
    else if (expected > 0) expect(result).toBeGreaterThan(0);
    else expect(result).toBeLessThan(0);
  });
});

interface ReconcileScriptedStep {
  /** Status returned from the next `refresh()` call. */
  running: boolean;
  version?: string;
}

/**
 * Build a DaemonSession whose initial cached status is fixed
 * and whose subsequent refreshes return values from *script*.
 */
function scriptedSession(
  initial: { running: boolean; version?: string } | null,
  script: ReconcileScriptedStep[],
): DaemonSession {
  const session = new DaemonSession();
  if (initial !== null) {
    // @ts-expect-error  —  write-through to private cache for tests.
    session.status = { running: initial.running, rpc: "ipc:///tmp/test.rpc", pub: null, version: initial.version };
  }
  let step = 0;
  // @ts-expect-error  —  override the CLI shell-out for test.
  session.refresh = async () => {
    const next = script[step++];
    if (next === undefined) {
      // @ts-expect-error  —  write-through for tests.
      session.status = null;
      return null;
    }
    const status = {
      running: next.running,
      rpc: "ipc:///tmp/test.rpc",
      pub: null,
      version: next.version,
    };
    // @ts-expect-error  —  write-through for tests.
    session.status = next.running ? status : null;
    return next.running ? status : null;
  };
  return session;
}

function recordingDeps(): { deps: ReconcileDeps; calls: Array<"start" | "stop"> } {
  const calls: Array<"start" | "stop"> = [];
  return {
    calls,
    deps: {
      execDaemon: async (sub) => {
        calls.push(sub);
      },
    },
  };
}

function failingDeps(whichFails: "start" | "stop"): ReconcileDeps {
  return {
    execDaemon: async (sub) => {
      if (sub === whichFails) throw new Error(`simulated ${sub} failure`);
    },
  };
}

describe("DaemonSession.reconcile", () => {
  test("daemon not running -> started", async () => {
    const session = scriptedSession(null, [{ running: true, version: "2026.4.0" }]);
    const { deps, calls } = recordingDeps();
    const result = await session.reconcile("2026.4.0", deps);
    expect(result.outcome).toBe("started");
    expect(result.newVersion).toBe("2026.4.0");
    expect(calls).toEqual(["start"]);
  });

  test("daemon not running, start fails -> failed", async () => {
    const session = scriptedSession(null, []);
    const result = await session.reconcile("2026.4.0", failingDeps("start"));
    expect(result.outcome).toBe("failed");
  });

  test("start failure carries the reason as detail", async () => {
    const session = scriptedSession(null, []);
    const result = await session.reconcile("2026.4.0", failingDeps("start"));
    expect(result.detail).toContain("simulated start failure");
  });

  test("versions match -> up_to_date", async () => {
    const session = scriptedSession({ running: true, version: "2026.4.0" }, []);
    const { deps, calls } = recordingDeps();
    const result = await session.reconcile("2026.4.0", deps);
    expect(result.outcome).toBe("up_to_date");
    expect(calls).toEqual([]);
  });

  test("extension older than daemon -> older_client (no op)", async () => {
    const session = scriptedSession({ running: true, version: "2026.5.0" }, []);
    const { deps, calls } = recordingDeps();
    const result = await session.reconcile("2026.4.0", deps);
    expect(result.outcome).toBe("older_client");
    expect(result.newVersion).toBe("2026.5.0");
    expect(calls).toEqual([]);
  });

  test("extension newer than daemon -> restarted", async () => {
    const session = scriptedSession({ running: true, version: "2026.4.0" }, [{ running: true, version: "2026.5.0" }]);
    const { deps, calls } = recordingDeps();
    const result = await session.reconcile("2026.5.0", deps);
    expect(result.outcome).toBe("restarted");
    expect(result.previousVersion).toBe("2026.4.0");
    expect(result.newVersion).toBe("2026.5.0");
    expect(calls).toEqual(["stop", "start"]);
  });

  test("extension newer, stop fails -> failed", async () => {
    const session = scriptedSession({ running: true, version: "2026.4.0" }, []);
    const result = await session.reconcile("2026.5.0", failingDeps("stop"));
    expect(result.outcome).toBe("failed");
  });

  test("extension newer, restart succeeds but refresh shows not running -> failed", async () => {
    const session = scriptedSession({ running: true, version: "2026.4.0" }, [{ running: false }]);
    const { deps } = recordingDeps();
    const result = await session.reconcile("2026.5.0", deps);
    expect(result.outcome).toBe("failed");
  });
});
