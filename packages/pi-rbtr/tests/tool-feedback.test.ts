/**
 * Execute-level tests for the read tools' empty-result feedback.
 *
 * Drives the real `execute` closures with a captured `pi` and a mocked
 * `DaemonSession`, so we exercise the actual tool behaviour (not just
 * the `echoArgs` helper): when a call returns nothing, the arguments it
 * received are echoed back so a malformed argument is visible in context.
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

async function runTool(name: string, params: Record<string, unknown>): Promise<string> {
  const tool = registeredTools().get(name);
  if (!tool) throw new Error(`tool ${name} not registered`);
  const result = await tool.execute("id", params, signal, noop, ctx);
  return result.content.map((p) => p.text).join("");
}

describe("read_symbol empty-result feedback", () => {
  test("echoes a malformed file_paths so the model sees it in context", async () => {
    sendMock.mockResolvedValueOnce({ kind: "read_symbol", chunks: [] });
    const text = await runTool("rbtr_read_symbol", {
      symbol: "build_index",
      file_paths: ['["src/a.py"]'], // double-encoded, as pi delivers it
    });
    expect(text).toContain("Symbol not found: build_index");
    expect(text).toContain("Arguments received: file_paths=");
    expect(text).toContain("src/a.py");
  });

  test("no echo line when no optional args were passed", async () => {
    sendMock.mockResolvedValueOnce({ kind: "read_symbol", chunks: [] });
    const text = await runTool("rbtr_read_symbol", { symbol: "build_index" });
    expect(text).toBe("Symbol not found: build_index");
  });
});

describe("search / find_refs empty-result feedback", () => {
  test("search echoes keywords on no results", async () => {
    sendMock.mockResolvedValueOnce({ kind: "search", results: [] });
    const text = await runTool("rbtr_search", { query: "x", keywords: ['["a","b"]'] });
    expect(text).toContain("No results found.");
    expect(text).toContain("Arguments received: query=");
    expect(text).toContain("keywords=");
  });
});

describe("rbtr_find_refs reply", () => {
  const resolved = { sha: "a".repeat(40), source: "head" };
  const ref = {
    name: "from config import load_config",
    kind: "import",
    file_path: "src/app.py",
    line_start: 1,
    edge: "imports",
  };

  test.each([
    ["references", { kind: "find_refs", refs: [ref], resolved, file_paths: null }],
    ["no references", { kind: "find_refs", refs: [], resolved, file_paths: ["src/a.py"] }],
  ])("with %s, is the daemon's response as JSON, matching the tool's output schema", async (_name, response) => {
    sendMock.mockResolvedValueOnce(response);
    const tool = registeredTools().get("rbtr_find_refs");
    if (!tool) throw new Error("rbtr_find_refs not registered");
    const result = await tool.execute("id", { symbol: "load_config" }, signal, noop, ctx);
    expect(JSON.parse(result.content.map((p) => p.text).join(""))).toEqual(response);
    expect(result.structuredContent).toEqual(response);
    expect(Check(tool.outputSchema, result.structuredContent)).toBe(true);
  });

  test("its output schema rejects a response without the snapshot it read", () => {
    const tool = registeredTools().get("rbtr_find_refs");
    expect(Check(tool?.outputSchema ?? {}, { kind: "find_refs", refs: [], file_paths: null })).toBe(false);
  });
});
