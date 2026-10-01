/**
 * The `tool_result` hook that appends index facts, driven through the real
 * extension with the daemon's replies mocked.
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";

const { sendMock } = vi.hoisted(() => ({ sendMock: vi.fn() }));

vi.mock("../extensions/rbtr/daemon-session.js", () => ({
  DaemonSession: class {
    get available() {
      return true;
    }
    send = sendMock;
  },
  DaemonUnavailableError: class extends Error {},
}));

import { HEADER } from "../extensions/rbtr/annotate.js";
import rbtrIndexExtension from "../extensions/rbtr/index.js";

type Handler = (event: Record<string, unknown>, ctx: { cwd: string }) => Promise<unknown>;

function toolResultHook(): Handler {
  const handlers = new Map<string, Handler>();
  const pi = {
    on: (name: string, handler: Handler) => handlers.set(name, handler),
    registerCommand: () => {},
    registerTool: () => {},
  } as unknown as Parameters<typeof rbtrIndexExtension>[0];
  rbtrIndexExtension(pi);
  const hook = handlers.get("tool_result");
  if (!hook) throw new Error("no tool_result hook");
  return hook;
}

const grepResult = {
  type: "tool_result",
  toolName: "bash",
  toolCallId: "c",
  input: { command: "grep -rn union django" },
  content: [{ type: "text", text: "django/db/models/sql/query.py:12: from x import union" }],
  details: { exitCode: 0 },
  structuredContent: { output: "django/db/models/sql/query.py:12: from x import union", exit_code: 0 },
  isError: false,
};

const definition = {
  name: "QuerySet.union",
  kind: "method",
  file_path: "django/db/models/query.py",
  content: "",
  line_start: 939,
  line_end: 950,
};

describe("tool_result hook", () => {
  test("appends the facts to a bash search, keeping the output, the details and the structured result", async () => {
    sendMock
      .mockResolvedValueOnce({ kind: "read_symbol", chunks: [definition] })
      .mockResolvedValueOnce({ kind: "find_refs", refs: [] });
    const result = (await toolResultHook()(grepResult, { cwd: "/repo" })) as {
      content: Array<{ text: string }>;
      details: unknown;
      structuredContent: unknown;
    };
    expect(result.content[0]).toEqual(grepResult.content[0]);
    expect(result.content[1].text).toContain(HEADER);
    expect(result.content[1].text).toContain("defined: django/db/models/query.py:939 method QuerySet.union");
    expect(result.details).toEqual({ exitCode: 0 });
    expect(result.structuredContent).toEqual(grepResult.structuredContent);
  });

  test.each([
    ["a failed tool", { isError: true }],
    ["a call another tool made", { toolCallId: "p/1", parentToolCallId: "p" }],
  ])("leaves the output of %s alone", async (_name, change) => {
    sendMock.mockImplementation(async (request: { kind: string }) =>
      request.kind === "read_symbol" ? { kind: "read_symbol", chunks: [definition] } : { kind: "find_refs", refs: [] },
    );
    expect(await toolResultHook()({ ...grepResult, ...change }, { cwd: "/repo" })).toBeUndefined();
    sendMock.mockReset();
  });

  test("still outlines a file for the model after a script read it", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "rbtr-hook-"));
    writeFileSync(join(cwd, "big.py"), "x = 1\n".repeat(300));
    sendMock.mockResolvedValue({ kind: "list_symbols", chunks: [{ ...definition, file_path: "big.py" }] });
    const hook = toolResultHook();
    const read = { ...grepResult, toolName: "read", input: { path: "big.py" } };
    await hook({ ...read, toolCallId: "p/1", parentToolCallId: "p" }, { cwd });
    const result = (await hook(read, { cwd })) as { content: Array<{ text: string }> };
    expect(result.content[1].text).toContain("outline of big.py:");
    sendMock.mockReset();
  });

  test("leaves the output alone when the daemon fails", async () => {
    sendMock.mockRejectedValueOnce(new Error("socket closed"));
    expect(await toolResultHook()(grepResult, { cwd: "/repo" })).toBeUndefined();
  });

  test("leaves the output alone when the index knows nothing", async () => {
    sendMock.mockResolvedValueOnce({ kind: "read_symbol", chunks: [] });
    expect(await toolResultHook()(grepResult, { cwd: "/repo" })).toBeUndefined();
  });
});
