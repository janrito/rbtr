/**
 * Tests for the pure daemon-failure classifier in `classify.ts`.  No pi,
 * daemon or CLI involved; startup's decisions are tested through sessions
 * in `session-startup.test.ts`.
 */

import { describe, expect, test } from "vitest";

import { classifyDaemonFailure } from "../extensions/rbtr/classify.js";

describe("classifyDaemonFailure", () => {
  const locked = { kind: "error", code: "index_locked", message: "Another process holds the index lock" } as const;
  const other = { kind: "error", code: "internal", message: "Daemon failed to start within 5 s." } as const;

  test.each([
    [1, locked, "", "db-locked"],
    [1, other, "", "transient"],
    [1, null, "", "transient"],
  ] as const)("code=%i reply=%j stderr=%j -> %s", (code, reply, stderr, kind) => {
    expect(classifyDaemonFailure(code, reply, stderr).kind).toBe(kind);
  });

  test("transient failure carries rbtr's message, or stderr when it printed none", () => {
    expect(classifyDaemonFailure(1, other, "").message).toContain("Daemon failed to start");
    expect(classifyDaemonFailure(1, null, "Traceback …").message).toContain("Traceback");
  });
});
