/**
 * Index facts appended to `bash` and `read` output (extensions/rbtr/annotate.ts).
 */

import { describe, expect, test } from "vitest";

import {
  block,
  HEADER,
  type IndexLookups,
  readFacts,
  searchedName,
  searchedPhrase,
  searchFacts,
  searchPattern,
  shellWords,
} from "../extensions/rbtr/annotate.js";

describe("shellWords", () => {
  test("keeps quoted words whole and splits operators off", () => {
    expect(shellWords(`cd a && grep -rn "def union" django | head -3`)).toEqual([
      "cd",
      "a",
      "&&",
      "grep",
      "-rn",
      "def union",
      "django",
      "|",
      "head",
      "-3",
    ]);
  });

  test("honours single quotes and backslashes", () => {
    expect(shellWords(`rg 'QuerySet\\.union' x\\ y`)).toEqual(["rg", "QuerySet\\.union", "x y"]);
  });
});

describe("searchPattern", () => {
  test.each([
    [`grep -rn "def union" django | head -3`, "def union"],
    [`rg -n --type py 'QuerySet\\.union'`, "QuerySet\\.union"],
    [`git grep -n union`, "union"],
    [`cat f | grep -e combinator`, "combinator"],
    [`cd /repo && grep -rn --include=*.py -A 3 union .`, "union"],
    [`grep -- -weird file`, "-weird"],
  ])("finds the pattern in %s", (command, pattern) => {
    expect(searchPattern(command)).toBe(pattern);
  });

  test.each([[`ls -la`], [`python -m pytest`], [`echo grep`]])("finds none in %s", (command) => {
    expect(searchPattern(command)).toBeNull();
  });
});

describe("searchedName", () => {
  test.each([
    ["union", "union"],
    ["QuerySet.union", "QuerySet.union"],
    ["\\bunion\\b", "union"],
    ["def union", "union"],
    ["class QuerySet", "QuerySet"],
  ])("reads %s as the name %s", (pattern, name) => {
    expect(searchedName(pattern)).toBe(name);
  });

  test.each([["un"], ["ORDER BY term"], ["union.*order"]])("reads %s as no name", (pattern) => {
    expect(searchedName(pattern)).toBeNull();
  });
});

describe("searchedPhrase", () => {
  test("keeps the words of a phrase, dropping regex syntax", () => {
    expect(searchedPhrase("ORDER BY.*term")).toBe("ORDER BY term");
  });

  test("a single word is not a phrase", () => {
    expect(searchedPhrase("union")).toBeNull();
  });
});

function lookups(overrides: Partial<IndexLookups>): IndexLookups {
  return {
    readSymbol: async () => ({ kind: "read_symbol", chunks: [] }),
    findRefs: async () => ({ kind: "find_refs", refs: [] }),
    search: async () => ({ kind: "search", results: [] }),
    listSymbols: async () => ({ kind: "list_symbols", chunks: [] }),
    ...overrides,
  };
}

const union = {
  name: "QuerySet.union",
  kind: "method" as const,
  file_path: "django/db/models/query.py",
  content: "def union(self): ...",
  line_start: 939,
  line_end: 950,
};

describe("searchFacts", () => {
  test("a searched name gets its definition and what uses it", async () => {
    const index = lookups({
      readSymbol: async () => ({ kind: "read_symbol", chunks: [union] }),
      findRefs: async () => ({
        kind: "find_refs",
        refs: [
          {
            name: "compiler",
            kind: "module",
            file_path: "django/db/models/sql/compiler.py",
            line_start: 1,
            edge: "imports",
          },
          {
            name: "compiler",
            kind: "module",
            file_path: "django/db/models/sql/compiler.py",
            line_start: 9,
            edge: "imports",
          },
        ],
      }),
    });
    expect(await searchFacts(index, `grep -rn "def union" django`, "")).toEqual([
      "defined: django/db/models/query.py:939 method QuerySet.union",
      "used by: django/db/models/sql/compiler.py",
    ]);
  });

  test("a definition the output already shows is not repeated", async () => {
    const index = lookups({ readSymbol: async () => ({ kind: "read_symbol", chunks: [union] }) });
    const output = "django/db/models/query.py:939:    def union(self, *other_qs, all=False):";
    expect(await searchFacts(index, "grep -rn union django", output)).toEqual([]);
  });

  test("a searched phrase gets the index's best matches", async () => {
    const index = lookups({
      search: async () => ({
        kind: "search",
        results: [
          {
            name: "SQLCompiler.get_order_by",
            kind: "method",
            file_paths: ["django/db/models/sql/compiler.py"],
            preview: { text: "", clipped: false, total_lines: 1 },
            line_start: 254,
            line_end: 380,
            score: 1,
          },
        ],
      }),
    });
    expect(await searchFacts(index, `grep -rn "ORDER BY term" django`, "")).toEqual([
      "match: django/db/models/sql/compiler.py:254 method SQLCompiler.get_order_by",
    ]);
  });

  test("a command that is not a search gets nothing", async () => {
    expect(await searchFacts(lookups({}), "python -m pytest", "")).toEqual([]);
  });
});

describe("readFacts", () => {
  const symbols = Array.from({ length: 12 }, (_, i) => ({
    name: `f${i}`,
    kind: "function" as const,
    file_path: "big.py",
    line_start: i * 10 + 1,
    line_end: i * 10 + 9,
  }));
  const index = lookups({ listSymbols: async () => ({ kind: "list_symbols", chunks: symbols }) });

  test("a long file gets an outline, capped, with the rest counted", async () => {
    const lines = await readFacts(index, "big.py", 500);
    expect(lines[0]).toBe("outline of big.py:");
    expect(lines[1]).toBe("  f0 (function) lines 1-9");
    expect(lines.at(-1)).toBe("  +4 more: rbtr_list_symbols");
    expect(lines).toHaveLength(10);
  });

  test("a short file gets none", async () => {
    expect(await readFacts(index, "big.py", 150)).toEqual([]);
  });
});

describe("block", () => {
  test("heads the lines and keeps at most ten", () => {
    const text = block(Array.from({ length: 14 }, (_, i) => `line ${i}`));
    expect(text?.split("\n")[0]).toBe(HEADER);
    expect(text?.split("\n")).toHaveLength(11);
  });

  test("is nothing when there are no lines", () => {
    expect(block([])).toBeNull();
  });
});
