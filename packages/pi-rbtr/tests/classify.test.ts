/**
 * Tests for the pure daemon-failure / session-start
 * classifiers in `classify.ts`.  No pi, daemon or CLI involved.
 */

import { describe, expect, test } from "vitest";

import { classifyDaemonFailure, decideStartupDecision } from "../extensions/rbtr/classify.js";
import type { StatusResponse } from "../extensions/rbtr/generated/protocol.js";

describe("classifyDaemonFailure", () => {
  const locked = { kind: "error", code: "index_locked", message: "Another process holds the index lock" } as const;
  const other = { kind: "error", code: "internal", message: "Daemon failed to start within 5 s." } as const;

  test.each([
    [127, null, "rbtr: command not found", "missing-cli"],
    [1, null, "spawn rbtr ENOENT", "missing-cli"],
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

  test("missing-cli message gives install instructions", () => {
    expect(classifyDaemonFailure(127, null, "command not found").message).toContain("uv tool install rbtr");
  });
});

function statusWithRefs(count: number): StatusResponse {
  return {
    kind: "status",
    db_path: "/tmp/test/index.duckdb",
    indexed_refs: Array.from({ length: count }, (_, i) => ({
      sha: `${i}`.repeat(40),
      names: [],
      total: 10,
      embedded: 10,
    })),
    watched: [],
    active_build: null,
    active_embed: null,
  } as StatusResponse;
}

describe("decideStartupDecision", () => {
  test("null status with unresolved CLI -> missing-cli", () => {
    expect(decideStartupDecision(false, null, true)).toEqual({ kind: "missing-cli" });
  });

  test("null status with resolved CLI -> transient, indexes when autoIndex", () => {
    expect(decideStartupDecision(true, null, true)).toEqual({ kind: "transient", index: true });
  });

  test("null status with resolved CLI, autoIndex off -> transient, no index", () => {
    expect(decideStartupDecision(true, null, false)).toEqual({ kind: "transient", index: false });
  });

  test("status with refs -> indexed", () => {
    expect(decideStartupDecision(true, statusWithRefs(1), true)).toEqual({ kind: "indexed" });
  });

  test("status with no refs -> empty, indexes when autoIndex", () => {
    expect(decideStartupDecision(true, statusWithRefs(0), true)).toEqual({ kind: "empty", index: true });
  });
});
