/**
 * The `tool_result` hook that appends index facts, driven through the real
 * extension with the daemon's replies mocked.
 */

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
  test("appends the facts to a bash search, keeping the output and the details", async () => {
    sendMock
      .mockResolvedValueOnce({ kind: "read_symbol", chunks: [definition] })
      .mockResolvedValueOnce({ kind: "find_refs", refs: [] });
    const result = (await toolResultHook()(grepResult, { cwd: "/repo" })) as {
      content: Array<{ text: string }>;
      details: unknown;
    };
    expect(result.content[0]).toEqual(grepResult.content[0]);
    expect(result.content[1].text).toContain(HEADER);
    expect(result.content[1].text).toContain("defined: django/db/models/query.py:939 method QuerySet.union");
    expect(result.details).toEqual({ exitCode: 0 });
  });

  test("leaves a failed tool's output alone", async () => {
    expect(await toolResultHook()({ ...grepResult, isError: true }, { cwd: "/repo" })).toBeUndefined();
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
