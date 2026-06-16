/**
 * Execute-level tests for the read tools' empty-result feedback.
 *
 * Drives the real `execute` closures with a captured `pi` and a mocked
 * `DaemonSession`, so we exercise the actual tool behaviour (not just
 * the `echoArgs` helper): when a call returns nothing, the arguments it
 * received are echoed back so a malformed argument is visible in context.
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

import rbtrIndexExtension from "../extensions/rbtr/index.js";

interface ToolDef {
  name: string;
  execute: (
    id: string,
    params: Record<string, unknown>,
    signal: AbortSignal,
    onUpdate: () => void,
    ctx: { cwd: string },
  ) => Promise<{ content: Array<{ type: string; text: string }>; details: Record<string, unknown> }>;
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

  test("find_refs echoes file_paths on no results", async () => {
    sendMock.mockResolvedValueOnce({ kind: "find_refs", refs: [] });
    const text = await runTool("rbtr_find_refs", { symbol: "load_config", file_paths: ['["src/a.py"]'] });
    expect(text).toContain("No references found for: load_config");
    expect(text).toContain("Arguments received: file_paths=");
  });
});
