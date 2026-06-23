/**
 * Tests for ``extractPayload`` — the single chokepoint both transports
 * pass through.  The daemon path carries the response on
 * ``details.response``; the CLI fallback prints one JSON response object
 * to stdout, captured as the tool content text.  Both must yield the
 * same list items, so the renderers don't care which transport ran.
 */

import { describe, expect, test } from "vitest";

import type { SearchHitOut } from "../extensions/rbtr/generated/protocol.js";
import { extractPayload, fileScopeSuffix, formatWatched } from "../extensions/rbtr/render.js";

// Minimal theme: styling is identity so assertions see raw text.
const plainTheme = {
  fg: (_style: string, text: string) => text,
  bold: (text: string) => text,
} as unknown as Parameters<typeof fileScopeSuffix>[1];

type ToolResult = Parameters<typeof extractPayload>[0];

const hit: SearchHitOut = {
  name: "load_config",
  kind: "function",
  file_path: "src/config.py",
  content: "def load_config(): ...",
  line_start: 1,
  line_end: 3,
  score: 0.9,
};

function daemonResult(response: unknown): ToolResult {
  return { content: [], details: { fromDaemon: true, response } } as unknown as ToolResult;
}

function cliResult(stdout: string): ToolResult {
  return {
    content: [{ type: "text", text: stdout }],
    details: { fromCli: true },
  } as unknown as ToolResult;
}

describe("extractPayload", () => {
  test("reads the list field off the daemon response", () => {
    const result = daemonResult({ kind: "search", results: [hit] });
    expect(extractPayload<SearchHitOut>(result, "search", "results")).toEqual([hit]);
  });

  test("parses the CLI fallback's single response object", () => {
    const result = cliResult(JSON.stringify({ kind: "search", results: [hit] }));
    expect(extractPayload<SearchHitOut>(result, "search", "results")).toEqual([hit]);
  });

  test("returns empty when the response kind does not match", () => {
    const result = cliResult(JSON.stringify({ kind: "read_symbol", chunks: [] }));
    expect(extractPayload<SearchHitOut>(result, "search", "results")).toEqual([]);
  });

  test("returns empty for unparseable content", () => {
    expect(extractPayload<SearchHitOut>(cliResult("not json"), "search", "results")).toEqual([]);
  });
});

describe("fileScopeSuffix", () => {
  test("is empty when no file_paths are given", () => {
    expect(fileScopeSuffix({ symbol: "x" }, plainTheme)).toBe("");
    expect(fileScopeSuffix({ file_paths: [] }, plainTheme)).toBe("");
  });

  test("names the single scoped path (shortened)", () => {
    expect(fileScopeSuffix({ file_paths: ["packages/rbtr/src/rbtr/index/search.py"] }, plainTheme)).toBe(
      " in rbtr/index/search.py",
    );
  });

  test("counts multiple scoped paths", () => {
    expect(fileScopeSuffix({ file_paths: ["a.py", "b.py", "c.py"] }, plainTheme)).toBe(" in 3 files");
  });

  test("ignores non-string entries", () => {
    expect(fileScopeSuffix({ file_paths: [42, null] }, plainTheme)).toBe("");
  });

  test("decodes a bare JSON-string file_paths (provider double-encoding)", () => {
    expect(fileScopeSuffix({ file_paths: '["packages/rbtr/src/rbtr/index/search.py"]' }, plainTheme)).toBe(
      " in rbtr/index/search.py",
    );
  });

  test("decodes a one-element list wrapping a JSON string", () => {
    expect(fileScopeSuffix({ file_paths: ['["a.py", "b.py"]'] }, plainTheme)).toBe(" in 2 files");
  });
});

describe("formatWatched", () => {
  test("renders indexed / pending / unresolvable markers", () => {
    const out = formatWatched([
      { ref: "main", sha: "a".repeat(40), indexed: true },
      { ref: "feature", sha: "b".repeat(40), indexed: false },
      { ref: "deleted", sha: null, indexed: false },
    ]).join("\n");
    expect(out).toContain("Watching:");
    expect(out).toContain("✓ main");
    expect(out).toContain("⟳ feature");
    expect(out).toContain("✗ deleted");
    expect(out).toContain("unresolvable");
  });

  test("empty watch set renders nothing", () => {
    expect(formatWatched([])).toEqual([]);
  });
});
