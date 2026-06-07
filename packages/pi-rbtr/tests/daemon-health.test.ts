/**
 * Tests for {@link DaemonSession.detectTransition}.
 *
 * Reuses the `scriptedSession` pattern from
 * `daemon-reconcile.test.ts`: write-through to private
 * `session.status`, override `refresh()` to return scripted
 * results, call `detectTransition()`, assert the variant.
 *
 * No ZMQ, no fake daemon — pure state-transition logic.
 */

import { describe, expect, test } from "vitest";

import { DaemonSession } from "../extensions/rbtr/daemon-session.js";
import { transitionScenarios } from "./daemon-health.cases.js";

interface ScriptedStep {
  running: boolean;
  version?: string;
}

/**
 * Build a DaemonSession with a fixed initial `available` state
 * and a scripted `refresh()` that sets the next state.
 */
function scriptedSession(initial: { running: boolean } | null, script: ScriptedStep[]): DaemonSession {
  const session = new DaemonSession();
  if (initial !== null) {
    // @ts-expect-error  —  write-through to private cache for tests.
    session.status = initial.running ? { running: true, rpc: "ipc:///tmp/test.rpc", pub: null } : null;
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

describe("DaemonSession.detectTransition", () => {
  test.each(transitionScenarios)("$name", async ({ initialAvailable, refreshedAvailable, expectedKind }) => {
    const session = scriptedSession(initialAvailable ? { running: true } : null, [{ running: refreshedAvailable }]);

    const result = await session.detectTransition();
    expect(result.kind).toBe(expectedKind);
  });
});
