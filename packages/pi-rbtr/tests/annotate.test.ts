/**
 * How a shell command is read for the `[rbtr index]` note: its words, its
 * search pattern, whether it searches outside the repository, and the name
 * or phrase it searches for; and the facts a note holds, from a stand-in
 * index. How a session gets the notes is tested in `session-notes.test.ts`.
 */

import { describe, expect, test } from "vitest";

import {
  type IndexLookups,
  readFacts,
  searchedName,
  searchedPhrase,
  searchesOutside,
  searchFacts,
  searchPattern,
  shellWords,
} from "../extensions/rbtr/annotate.js";
import type { ChunkKind, SearchHitOut, SymbolOut } from "../extensions/rbtr/generated/protocol.js";

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
    [`cat f | grep -e -combinator`, "-combinator"],
    [`grep -rn src -e union`, "union"],
    [`cd /repo && grep -rn --include=*.py -A 3 union .`, "union"],
    [`grep -- -weird file`, "-weird"],
  ])("finds the pattern in %s", (command, pattern) => {
    expect(searchPattern(command)).toBe(pattern);
  });

  test.each([[`ls src`], [`python run.py`], [`echo grep`]])("finds none in %s", (command) => {
    expect(searchPattern(command)).toBeNull();
  });
});

describe("searchesOutside", () => {
  test.each([
    ["cd /tmp && grep -rn foo ."],
    ["grep -n rbtr ~/.pi/agent/settings.json"],
    ["cd /opt/homebrew/lib && grep -n prepareArguments types.d.ts"],
    ["rg -n foo /etc /var"],
    ["grep -rn /tmp -e foo"],
  ])("%s searches only outside the repository", (command) => {
    expect(searchesOutside(command, "/repo", "/repo")).toBe(true);
  });

  test.each([
    ["grep -rn foo src"],
    ["grep -rn foo"],
    ["cd src && rg foo ."],
    ["grep -rn foo /repo/src /tmp"],
    ["git grep -n foo -- src"],
    ["ls"],
  ])("%s does not search only outside the repository", (command) => {
    expect(searchesOutside(command, "/repo", "/repo")).toBe(false);
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

/** A chunk of the stand-in index: *kind* *name* at *file_path*:*line_start*. */
const chunk = (kind: ChunkKind, name: string, file_path: string, line_start: number): SymbolOut => ({
  name,
  kind,
  file_path,
  content: "",
  line_start,
  line_end: line_start + 1,
});

const LONG_NAME = "n".repeat(81);

/**
 * The stand-in index: every name's chunks, the files using each name, the
 * four matches of any search, each a doc section named for the query, and
 * the outline of any file.
 */
const INDEX = {
  chunks: [
    chunk("function", "load_config", "app/config.py", 17),
    chunk("config_key", "timeout", "settings.json", 3),
    chunk("function", "timeout", "app/net.py", 5),
    ...["settings.json", "deploy/dev.json", "deploy/prod.json", "deploy/test.json"].map((file) =>
      chunk("config_key", "retention_days", file, 2),
    ),
    chunk("function", "union", "app/sets.py", 1),
  ],
  users: {
    load_config: ["app/cli.py", "app/report.py"],
    timeout: ["app/client.py"],
    union: ["b.py", "c.py", "d.py", "e.py", "f.py", "g.py", "h.py"],
  } as Record<string, string[]>,
  matches: (query: string) =>
    ["app/a.py", "app/b.py", "app/c.py", "app/d.py"].map(
      (file): SearchHitOut => ({
        name: query,
        kind: "doc_section",
        file_paths: [file],
        preview: { text: "", clipped: false, total_lines: 1 },
        line_start: 10,
        line_end: 11,
        score: 1,
      }),
    ),
  outline: [
    chunk("import", "os", "app/store.py", 1),
    chunk("comment", "note", "app/store.py", 3),
    chunk("raw_chunk", "", "app/store.py", 4),
    chunk("class", LONG_NAME, "app/store.py", 10),
    ...Array.from({ length: 9 }, (_, i) => chunk("function", `step_${i}`, "app/store.py", 20 + 10 * i)),
  ],
};

const resolved = { sha: "a".repeat(40), source: "head" } as const;
const index: IndexLookups = {
  readSymbol: async (name) => ({
    kind: "read_symbol",
    chunks: INDEX.chunks.filter((c) => c.name === name),
    resolved,
    file_paths: null,
  }),
  findRefs: async (name) => ({
    kind: "find_refs",
    refs: (INDEX.users[name] ?? []).map((file_path) => ({
      name: "caller",
      kind: "function",
      file_path,
      line_start: 1,
      edge: "calls",
    })),
    resolved,
    file_paths: null,
  }),
  search: async (query) => ({ kind: "search", results: INDEX.matches(query), resolved }),
  listSymbols: async (file_path) => ({ kind: "list_symbols", chunks: INDEX.outline, resolved, file_path }),
};

describe("searchFacts", () => {
  test.for<[string, string, string, string[]]>([
    [
      "a name the output does not define gets its definition and its users",
      "grep -rln load_config app/cli.py",
      "app/cli.py\n",
      ["defined: app/config.py:17 function load_config", "used by: app/cli.py, app/report.py"],
    ],
    [
      "a definition the output shows is not repeated",
      'grep -rn "def load_config" app',
      "app/config.py:17:def load_config():\n",
      ["used by: app/cli.py, app/report.py"],
    ],
    [
      "a name a config file defines gets the code's definitions instead",
      "grep -rln timeout settings.json",
      "settings.json\n",
      ["defined: app/net.py:5 function timeout", "used by: app/client.py"],
    ],
    [
      "a name only config files define gets three of its keys",
      "grep -rln retention_days .",
      "settings.json\n",
      [
        "defined: settings.json:2 config_key retention_days",
        "defined: deploy/dev.json:2 config_key retention_days",
        "defined: deploy/prod.json:2 config_key retention_days",
      ],
    ],
    [
      "a name's users past five are counted, not listed",
      "grep -rln union b.py",
      "b.py\n",
      ["defined: app/sets.py:1 function union", "used by: b.py, c.py, d.py, e.py, f.py (+2 more)"],
    ],
    ["a name the index does not know gets none", "grep -rln parquets .", "", []],
    ["a word too short to be a name gets none", "grep -rn ab .", "", []],
    [
      "a phrase gets the index's top three matches",
      'grep -rn "Reads sampled queries" .',
      "",
      [
        "match: app/a.py:10 doc_section Reads sampled queries",
        "match: app/b.py:10 doc_section Reads sampled queries",
        "match: app/c.py:10 doc_section Reads sampled queries",
      ],
    ],
    ["a command that searches nothing gets none", "echo load_config", "load_config\n", []],
  ])("%s", async ([, command, output, expected]) => {
    expect(await searchFacts(index, command, output)).toEqual(expected);
  });
});

describe("readFacts", () => {
  test.for<[string, number, string[]]>([
    [
      "a long file gets eight of its symbols, names cut to 80 characters, without imports, comments or unnamed chunks",
      200,
      [
        "outline of app/store.py:",
        `  ${"n".repeat(79)}… (class) lines 10-11`,
        ...Array.from({ length: 7 }, (_, i) => `  step_${i} (function) lines ${20 + 10 * i}-${21 + 10 * i}`),
        "  +2 more: rbtr_list_symbols",
      ],
    ],
    ["a file shorter than 200 lines gets none", 199, []],
  ])("%s", async ([, lineCount, expected]) => {
    expect(await readFacts(index, "app/store.py", lineCount)).toEqual(expected);
  });
});
