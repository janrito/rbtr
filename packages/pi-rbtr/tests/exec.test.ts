/**
 * How `runRbtr` reports an `rbtr --json` command that failed without
 * printing its `ErrorResponse`: a crash, which the real CLI does not do on
 * demand. A printed `ErrorResponse` is tested through a session, in
 * `tool-replies.test.ts`, except running out of memory, which needs a CLI
 * whose DuckDB limit is too low to open the index.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { describe, expect, test } from "vitest";

import { runRbtr } from "../extensions/rbtr/exec.js";
import { RbtrReplyError } from "../extensions/rbtr/reply-error.js";

function piExiting(result: { stdout: string; stderr: string; code: number }): Pick<ExtensionAPI, "exec"> {
  return { exec: async () => ({ ...result, killed: false }) };
}

const rbtr = { executable: "rbtr", baseArgs: ["--json"], description: "rbtr (PATH)" };

describe("runRbtr on a non-zero exit", () => {
  test("throws stderr as a plain error when stdout holds no ErrorResponse", async () => {
    const run = runRbtr(piExiting({ stdout: "", stderr: "Traceback …", code: 1 }), rbtr, ["read-symbol", "x"]);
    await expect(run).rejects.not.toBeInstanceOf(RbtrReplyError);
    await expect(run).rejects.toThrow("Traceback …");
  });

  test("adds how to raise the memory limit to an out-of-memory ErrorResponse", async () => {
    const stdout = JSON.stringify({ kind: "error", code: "out_of_memory", message: "Out of Memory Error" });
    const run = runRbtr(piExiting({ stdout, stderr: "", code: 1 }), rbtr, ["gc"]);
    await expect(run).rejects.toThrow("Out of Memory Error\nRaise duckdb_memory_limit in config.toml");
  });
});
