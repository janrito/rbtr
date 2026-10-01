/**
 * Execute-level tests for the query tools' replies.
 *
 * Drives the real `execute` closures with a captured `pi` and a mocked
 * `DaemonSession`: every reply, empty or not, is rbtr's response as
 * JSON, carried as the structured result and matching the tool's
 * declared output schema.
 */

import { Check } from "typebox/schema";
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

import rbtrIndexExtension from "../extensions/rbtr/index.js";

interface ToolDef {
  name: string;
  outputSchema: Parameters<typeof Check>[0];
  execute: (
    id: string,
    params: Record<string, unknown>,
    signal: AbortSignal,
    onUpdate: () => void,
    ctx: { cwd: string },
  ) => Promise<{
    content: Array<{ type: string; text: string }>;
    details: Record<string, unknown>;
    structuredContent?: unknown;
  }>;
}

function registeredTools(): Map<string, ToolDef> {
  const tools = new Map<string, ToolDef>();
  const pi = {
    on: () => {},
    registerCommand: () => {},
    registerTool: (def: ToolDef) => tools.set(def.name, def),
  } as unknown as Parameters<typeof rbtrIndexExtension>[0];
  rbtrIndexExtension(pi);
  return tools;
}

const ctx = { cwd: "/repo" };
const noop = () => {};
const signal = new AbortController().signal;

const resolved = { sha: "a".repeat(40), source: "head" };
const ref = {
  name: "from config import load_config",
  kind: "import",
  file_path: "src/app.py",
  line_start: 1,
  edge: "imports",
};
const outline = {
  name: "load_config",
  kind: "function",
  file_path: "src/config.py",
  scope: "",
  language: "python",
  line_start: 1,
  line_end: 3,
};
const source = { ...outline, content: "def load_config(): ..." };
const hit = {
  name: "load_config",
  kind: "function",
  file_paths: ["src/config.py"],
  scope: "",
  language: "python",
  preview: { text: "def load_config(): ...", clipped: false, total_lines: 1 },
  line_start: 1,
  line_end: 1,
  score: 1,
};

describe.each([
  [
    "rbtr_search",
    { query: "load_config" },
    { kind: "search", results: [hit], resolved },
    { kind: "search", results: [], resolved: null },
  ],
  [
    "rbtr_read_symbol",
    { symbol: "load_config" },
    { kind: "read_symbol", chunks: [source], resolved, file_paths: null },
    { kind: "read_symbol", chunks: [], resolved, file_paths: ["src/a.py"] },
  ],
  [
    "rbtr_find_refs",
    { symbol: "load_config" },
    { kind: "find_refs", refs: [ref], resolved, file_paths: null },
    { kind: "find_refs", refs: [], resolved, file_paths: ["src/a.py"] },
  ],
  [
    "rbtr_list_symbols",
    { file: "src/config.py" },
    { kind: "list_symbols", chunks: [outline], resolved, file_path: "src/config.py" },
    { kind: "list_symbols", chunks: [], resolved, file_path: "nope.py" },
  ],
  [
    "rbtr_changed_symbols",
    { base: "main", head: "HEAD" },
    {
      kind: "changed_symbols",
      changes: [{ chunk: outline, change: "modified" }],
      base_sha: "b".repeat(40),
      head_sha: "c".repeat(40),
      file_paths: null,
    },
    { kind: "changed_symbols", changes: [], base_sha: "b".repeat(40), head_sha: "c".repeat(40), file_paths: null },
  ],
])("%s reply", (name, params, found, empty) => {
  test.each([
    ["results", found],
    ["no results", empty],
  ])("with %s, is the daemon's response as JSON, matching the tool's output schema", async (_case, response) => {
    sendMock.mockResolvedValueOnce(response);
    const tool = registeredTools().get(name);
    if (!tool) throw new Error(`${name} not registered`);
    const result = await tool.execute("id", params, signal, noop, ctx);
    expect(JSON.parse(result.content.map((p) => p.text).join(""))).toEqual(response);
    expect(result.structuredContent).toEqual(response);
    expect(Check(tool.outputSchema, result.structuredContent)).toBe(true);
  });

  test("its output schema rejects a response missing a field", () => {
    const { kind } = found;
    expect(Check(registeredTools().get(name)?.outputSchema ?? {}, { kind })).toBe(false);
  });
});
