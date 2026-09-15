/**
 * Tests for ``extractPayload`` — the single chokepoint both transports
 * pass through.  The daemon path carries the response on
 * ``details.response``; the CLI fallback prints one JSON response object
 * to stdout, captured as the tool content text.  Both must yield the
 * same list items, so the renderers don't care which transport ran.
 */

import { describe, expect, test } from "vitest";

import type { SearchHitOut, StatusResponse } from "../extensions/rbtr/generated/protocol.js";
import {
  extractPayload,
  fileScopeSuffix,
  footerLabel,
  formatWatched,
  renderSearchResult,
  renderStatusText,
} from "../extensions/rbtr/render.js";

// Minimal theme: styling is identity so assertions see raw text.
const plainTheme = {
  fg: (_style: string, text: string) => text,
  bold: (text: string) => text,
} as unknown as Parameters<typeof fileScopeSuffix>[1];

type ToolResult = Parameters<typeof extractPayload>[0];

const hit: SearchHitOut = {
  name: "load_config",
  kind: "function",
  file_paths: ["src/config.py"],
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

describe("renderStatusText", () => {
  // One response covering every embed state a ref can be in: fully
  // embedded, partly, not at all, and an empty snapshot.
  const status: StatusResponse = {
    kind: "status",
    db_path: "/db",
    indexed_refs: [
      { sha: "a".repeat(40), names: ["HEAD", "main"], total: 1200, embedded: 1200 },
      { sha: "b".repeat(40), names: [], total: 1200, embedded: 512 },
      { sha: "c".repeat(40), names: [], total: 1200, embedded: 0 },
      { sha: "d".repeat(40), names: [], total: 0, embedded: 0 },
    ],
    active_build: {
      repo_path: "/repo",
      ref: "e".repeat(40),
      phase: "chunking",
      current: 3,
      total: 10,
      elapsed_seconds: 65,
    },
    active_embed: { repo_path: "/repo", ref: "e".repeat(40), current: 5, total: 20, elapsed_seconds: 12 },
  };

  test("renders one line per ref, in the shape the TUI uses", () => {
    const lines = renderStatusText(status).split("\n");
    expect(lines).toContain("Index: 1.2k symbols (/db)");
    expect(lines).toContain("  aaaaaaaaaaaa (HEAD, main)  1.2k indexed  1.2k embedded ✓");
    expect(lines).toContain("  bbbbbbbbbbbb  1.2k indexed  512 embedded (43%)");
    expect(lines).toContain("  cccccccccccc  1.2k indexed  not embedded");
    expect(lines).toContain("  dddddddddddd  0 indexed  0 embedded ✓");
  });

  test("reports the running build and embed pass", () => {
    const lines = renderStatusText(status).split("\n");
    expect(lines).toContain("Building: eeeeeeeeeeee — chunking 3/10 (30%) — 1m05s");
    expect(lines).toContain("Embedding: eeeeeeeeeeee — 5/20 (25%) — 12s");
  });

  test("suppresses the percentage when the job has no total yet", () => {
    const starting: StatusResponse = {
      kind: "status",
      active_build: {
        repo_path: "/repo",
        ref: "f".repeat(40),
        phase: "walking",
        current: 0,
        total: 0,
        elapsed_seconds: 2,
      },
    };
    expect(renderStatusText(starting).split("\n")).toContain("Building: ffffffffffff — walking 0/0 — 2s");
  });

  test("pads the seconds on the minute boundary", () => {
    const justOverAMinute: StatusResponse = {
      kind: "status",
      active_embed: { repo_path: "/repo", ref: "f".repeat(40), current: 1, total: 2, elapsed_seconds: 60 },
    };
    expect(renderStatusText(justOverAMinute).split("\n")).toContain("Embedding: ffffffffffff — 1/2 (50%) — 1m00s");
  });
});

describe("footerLabel", () => {
  test("marks a fully embedded index with a filled glyph and no suffix", () => {
    expect(footerLabel({ total: 3500, embedded: 3500 }, true)).toBe("rbtr: ● 3.5k symbols");
  });

  test("reports the percentage while embedding is under way", () => {
    expect(footerLabel({ total: 3500, embedded: 1470 }, true)).toBe("rbtr: ○ 3.5k symbols · 42% embedded");
  });

  test("says so when nothing is embedded", () => {
    expect(footerLabel({ total: 3500, embedded: 0 }, true)).toBe("rbtr: ○ 3.5k symbols · not embedded");
  });

  test("names a missing daemon before the embed state", () => {
    expect(footerLabel({ total: 3500, embedded: 3500 }, false)).toBe("rbtr: ● 3.5k symbols · no daemon");
    expect(footerLabel({ total: 3500, embedded: 0 }, false)).toBe("rbtr: ○ 3.5k symbols · no daemon · not embedded");
  });
});

// Marks the `accent` role with guillemets so highlighted terms are
// visible in assertions; every other role is identity.
const markTheme = {
  fg: (style: string, text: string) => (style === "accent" ? `«${text}»` : text),
  bold: (text: string) => text,
} as unknown as Parameters<typeof renderSearchResult>[2];

function renderText(result: ToolResult, expanded: boolean): string {
  return renderSearchResult(result, { expanded, isPartial: false }, markTheme).render(1000).join("\n");
}

describe("renderSearchResult", () => {
  const previewHit: SearchHitOut = {
    name: "load_config",
    kind: "function",
    file_paths: ["src/config.py"],
    content: "def load_config(path):\n    return read(path)",
    line_start: 1,
    line_end: 2,
    score: 0.9,
    match_line_offset: 0,
    matched_terms: ["load_config", "config", "load"],
  };

  const plainHit: SearchHitOut = {
    name: "helper",
    kind: "function",
    file_paths: ["src/util.py"],
    content: "def helper():\n    return UNIQUE_BODY",
    line_start: 1,
    line_end: 2,
    score: 0.5,
  };

  const windowHit: SearchHitOut = {
    name: "big",
    kind: "function",
    file_paths: ["src/big.py"],
    content: "def big():\na = 1\nGAP = 2\nc = 3\nd = 4\nNEEDLE = 5\ne = 6\nf = 7",
    line_start: 1,
    line_end: 8,
    score: 0.9,
    match_line_offset: 5,
    matched_terms: ["needle"],
  };

  test("collapsed view surfaces the matched line, highlighted", () => {
    const out = renderText(daemonResult({ kind: "search", results: [previewHit] }), false);
    expect(out).toContain("«load_config»");
  });

  test("collapsed view stays header-only without a match offset", () => {
    const out = renderText(daemonResult({ kind: "search", results: [plainHit] }), false);
    expect(out).not.toContain("UNIQUE_BODY");
  });

  test("expanded view shows the signature, then scrolls to the highlighted match", () => {
    const out = renderText(daemonResult({ kind: "search", results: [windowHit] }), true);
    expect(out).toContain("def big()");
    expect(out).toContain("«NEEDLE»");
    expect(out).toContain("…");
    expect(out).not.toContain("GAP");
  });
});
