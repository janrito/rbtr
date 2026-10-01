/**
 * Which rbtr tools the model starts with, and how the rest become active.
 *
 * Drives the real extension with a captured `pi`, as tool-feedback.test.ts
 * does, adding pi's active-tool calls so the loader's effect is visible.
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

import { RbtrDaemonError } from "../extensions/rbtr/daemon-client.js";
import rbtrIndexExtension from "../extensions/rbtr/index.js";
import { startingTools } from "../extensions/rbtr/tool-set.js";

interface ToolDef {
  name: string;
  execute: (
    id: string,
    params: Record<string, unknown>,
    signal: AbortSignal,
    onUpdate: () => void,
    ctx: { cwd: string },
  ) => Promise<{ content: Array<{ type: string; text: string }> }>;
}

function extension(active: string[]) {
  const tools = new Map<string, ToolDef>();
  const setActiveTools = vi.fn();
  const pi = {
    on: () => {},
    registerCommand: () => {},
    registerTool: (def: ToolDef) => tools.set(def.name, def),
    getActiveTools: () => active,
    setActiveTools,
  } as unknown as Parameters<typeof rbtrIndexExtension>[0];
  rbtrIndexExtension(pi);
  return { tools, setActiveTools };
}

async function run(tool: ToolDef | undefined, params: Record<string, unknown>): Promise<string> {
  if (!tool) throw new Error("tool not registered");
  const result = await tool.execute("id", params, new AbortController().signal, () => {}, { cwd: "/repo" });
  return result.content.map((p) => p.text).join("");
}

const ALL = [
  "read",
  "bash",
  "edit",
  "write",
  "rbtr_search",
  "rbtr_read_symbol",
  "rbtr_find_refs",
  "rbtr_changed_symbols",
  "rbtr_list_symbols",
  "rbtr_watch",
  "rbtr_index_tools",
  "rbtr_status",
  "rbtr_gc",
];

describe("the starting tool set", () => {
  test("holds pi's tools, the tools that find code, rbtr_watch and the loader", () => {
    expect(startingTools(ALL)).toEqual(ALL.filter((n) => n !== "rbtr_status" && n !== "rbtr_gc"));
  });
});

describe("rbtr_index_tools", () => {
  test("makes rbtr_status and rbtr_gc active, and names them", async () => {
    const { tools, setActiveTools } = extension(startingTools(ALL));
    const text = await run(tools.get("rbtr_index_tools"), {});
    expect(setActiveTools).toHaveBeenCalledWith(expect.arrayContaining(["rbtr_status", "rbtr_gc", "rbtr_search"]));
    expect(text).toContain("rbtr_status");
    expect(text).toContain("rbtr_gc");
  });
});

describe("a ref that is not indexed", () => {
  test("the reply tells the model to call rbtr_watch", async () => {
    const { tools } = extension(startingTools(ALL));
    sendMock.mockRejectedValueOnce(
      new RbtrDaemonError({
        kind: "error",
        code: "index_not_built",
        message: "Ref 'main' is not indexed — run rbtr watch first",
      }),
    );
    const text = await run(tools.get("rbtr_changed_symbols"), { base: "main", head: "HEAD" });
    expect(text).toContain("Ref 'main' is not indexed");
    expect(text).toContain("Call rbtr_watch");
  });

  test("any other error gets no such hint", async () => {
    const { tools } = extension(startingTools(ALL));
    sendMock.mockRejectedValueOnce(new RbtrDaemonError({ kind: "error", code: "invalid_request", message: "bad" }));
    const text = await run(tools.get("rbtr_changed_symbols"), { base: "main", head: "HEAD" });
    expect(text).not.toContain("Call rbtr_watch");
  });
});
