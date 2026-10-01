/**
 * What pi-rbtr puts in front of the model: the tools that find code come
 * first, and the rules stay short enough to be read.
 */

import { describe, expect, test, vi } from "vitest";

vi.mock("../extensions/rbtr/daemon-session.js", () => ({
  DaemonSession: class {
    get available() {
      return true;
    }
    send = vi.fn();
  },
  DaemonUnavailableError: class extends Error {},
}));

import rbtrIndexExtension from "../extensions/rbtr/index.js";

interface ToolDef {
  name: string;
  description: string;
  promptSnippet: string;
  promptGuidelines: string[];
}

function registeredTools(): ToolDef[] {
  const tools: ToolDef[] = [];
  const pi = {
    on: () => {},
    registerCommand: () => {},
    registerTool: (def: ToolDef) => tools.push(def),
  } as unknown as Parameters<typeof rbtrIndexExtension>[0];
  rbtrIndexExtension(pi);
  return tools;
}

const FINDING = ["rbtr_search", "rbtr_read_symbol", "rbtr_find_refs", "rbtr_changed_symbols", "rbtr_list_symbols"];
const HOUSEKEEPING = ["rbtr_watch", "rbtr_index_tools", "rbtr_status", "rbtr_gc"];

describe("prompt text", () => {
  test("the tools that find code are registered before the housekeeping tools", () => {
    // pi lists tools in the prompt in the order they are registered.
    const names = registeredTools().map((t) => t.name);
    expect(names).toEqual([...FINDING, ...HOUSEKEEPING]);
  });

  test("the rules added to the system prompt number at most 16", () => {
    const count = registeredTools().reduce((n, t) => n + t.promptGuidelines.length, 0);
    expect(count).toBeLessThanOrEqual(16);
  });

  test("each tool that finds code says what it is used instead of", () => {
    for (const tool of registeredTools().filter((t) => FINDING.includes(t.name))) {
      expect(tool.description, tool.name).toMatch(/instead of/);
    }
  });
});
