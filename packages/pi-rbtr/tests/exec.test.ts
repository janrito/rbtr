/**
 * How `runRbtr` reports a failed `rbtr --json` command.
 */

import { describe, expect, test } from "vitest";

import { runRbtr } from "../extensions/rbtr/exec.js";
import { RbtrReplyError } from "../extensions/rbtr/reply-error.js";

function piExiting(result: { stdout: string; stderr: string; code: number }) {
  return { exec: async () => ({ ...result, killed: false }) } as unknown as Parameters<typeof runRbtr>[0];
}

const rbtr = { executable: "rbtr", baseArgs: ["--json"], description: "rbtr (PATH)" };

describe("runRbtr on a non-zero exit", () => {
  test("throws rbtr's ErrorResponse, printed on stdout, as an RbtrReplyError", async () => {
    const stdout = '{"kind":"error","code":"index_not_built","message":"Ref \'main\' is not indexed"}\n';
    const run = runRbtr(piExiting({ stdout, stderr: "", code: 1 }), rbtr, ["read-symbol", "x"]);
    await expect(run).rejects.toBeInstanceOf(RbtrReplyError);
    await expect(run).rejects.toMatchObject({ code: "index_not_built", message: "Ref 'main' is not indexed" });
  });

  test("throws stderr as a plain error when stdout holds no ErrorResponse", async () => {
    const run = runRbtr(piExiting({ stdout: "", stderr: "Traceback …", code: 1 }), rbtr, ["read-symbol", "x"]);
    await expect(run).rejects.not.toBeInstanceOf(RbtrReplyError);
    await expect(run).rejects.toThrow("Traceback …");
  });
});
