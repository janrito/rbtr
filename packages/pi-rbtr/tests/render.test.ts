/**
 * Tests for ``extractPayload`` — the single chokepoint both transports
 * pass through.  The daemon path carries the response on
 * ``details.response``; the CLI fallback prints one JSON response object
 * to stdout, captured as the tool content text.  Both must yield the
 * same list items, so the renderers don't care which transport ran.
 */

import { describe, expect, test } from "vitest";

import type { SearchHitOut } from "../extensions/rbtr/generated/protocol.js";
import { extractPayload } from "../extensions/rbtr/render.js";

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
