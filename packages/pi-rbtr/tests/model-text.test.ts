/**
 * The text the model reads for each rbtr reply, from fixed responses.
 * That every tool sends this text, and scripts the response, is tested
 * in `tool-replies.test.ts`.
 */

import { describe, expect, test } from "vitest";

import type { Response } from "../extensions/rbtr/generated/protocol.js";
import { modelText } from "../extensions/rbtr/model-text.js";

const head = { sha: "7d19de2e9c9a94658c36b55011b803a7991d0335", source: "head" } as const;

describe("modelText", () => {
  test.each<[string, Response, string]>([
    [
      "an outline names the file once, and each symbol with its scope",
      {
        kind: "list_symbols",
        file_path: "vllm/engine/arg_utils.py",
        resolved: head,
        chunks: [
          { name: "EngineArgs", kind: "class", file_path: "vllm/engine/arg_utils.py", line_start: 21, line_end: 600 },
          {
            name: "add_cli_args",
            kind: "method",
            file_path: "vllm/engine/arg_utils.py",
            scope: "EngineArgs",
            language: "python",
            line_start: 156,
            line_end: 560,
          },
        ],
      },
      [
        "read at 7d19de2e9c9a (head)",
        "outline of vllm/engine/arg_utils.py:",
        "  EngineArgs (class) lines 21-600",
        "  EngineArgs::add_cli_args (method) lines 156-560",
      ].join("\n"),
    ],
    [
      "a search hit gives where it is, its preview, and how much it leaves out",
      {
        kind: "search",
        resolved: head,
        results: [
          {
            name: "load_config",
            kind: "function",
            file_paths: ["src/config.py", "vendor/config.py"],
            preview: { text: "def load_config(path):\n    with open(path) as f:", clipped: true, total_lines: 5 },
            line_start: 3,
            line_end: 7,
            matched_terms: ["load_config"],
            match_line_offset: 0,
            score: 0.9,
          },
          {
            name: "MAX",
            kind: "variable",
            file_paths: ["src/config.py"],
            preview: { text: "MAX = 1", clipped: false, total_lines: 1 },
            line_start: 1,
            line_end: 1,
            score: 0.2,
          },
        ],
      },
      [
        "read at 7d19de2e9c9a (head): 2 result(s)",
        "match: src/config.py, vendor/config.py:3 function load_config",
        "  def load_config(path):",
        "      with open(path) as f:",
        "  … 3 more lines",
        "match: src/config.py:1 variable MAX",
        "  MAX = 1",
      ].join("\n"),
    ],
    [
      "a search of every repository names each hit's repository",
      {
        kind: "search",
        resolved: null,
        results: [
          {
            name: "f",
            kind: "function",
            file_paths: ["a.py"],
            repo_path: "/work/other/",
            preview: { text: "def f(): ...", clipped: false, total_lines: 1 },
            line_start: 1,
            line_end: 1,
            score: 0.5,
          },
        ],
      },
      "read at each repo's latest: 1 result(s)\nmatch: other/a.py:1 function f\n  def f(): ...",
    ],
    [
      "a read gives each definition's place, then its source",
      {
        kind: "read_symbol",
        resolved: head,
        file_paths: null,
        chunks: [
          {
            name: "add_cli_args",
            kind: "method",
            scope: "EngineArgs",
            file_path: "vllm/engine/arg_utils.py",
            content: "def add_cli_args(parser):\n    return parser",
            line_start: 156,
            line_end: 157,
          },
        ],
      },
      [
        "read at 7d19de2e9c9a (head)",
        "defined: vllm/engine/arg_utils.py:156-157 method EngineArgs::add_cli_args",
        "def add_cli_args(parser):",
        "    return parser",
      ].join("\n"),
    ],
    [
      "a read that finds nothing says so",
      { kind: "read_symbol", resolved: head, file_paths: null, chunks: [] },
      "read at 7d19de2e9c9a (head): no symbol by that name",
    ],
    [
      "references give how each one refers, and where",
      {
        kind: "find_refs",
        resolved: head,
        file_paths: null,
        refs: [
          {
            name: "from config import load_config",
            kind: "import",
            file_path: "src/app.py",
            line_start: 1,
            edge: "imports",
          },
        ],
      },
      "read at 7d19de2e9c9a (head): 1 ref(s)\nimports: src/app.py:1 import from config import load_config",
    ],
    [
      "changes give how each symbol changed, and where",
      {
        kind: "changed_symbols",
        base_sha: "a".repeat(40),
        head_sha: "b".repeat(40),
        file_paths: null,
        changes: [
          {
            change: "modified",
            chunk: { name: "m", kind: "method", scope: "C", file_path: "a.py", line_start: 4, line_end: 9 },
          },
        ],
      },
      "changes from aaaaaaaaaaaa to bbbbbbbbbbbb: 1 change(s)\nmodified: a.py:4 method C::m",
    ],
  ])("%s", (_label, response, expected) => {
    expect(modelText(response)).toBe(expected);
  });

  test("an error stays rbtr's JSON", () => {
    const error: Response = { kind: "error", code: "repo_not_found", message: "no such repo" };
    expect(JSON.parse(modelText(error))).toEqual(error);
  });
});
