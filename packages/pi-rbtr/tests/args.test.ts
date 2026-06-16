/**
 * Tests for echoArgs — the transparency helper that lets the model see
 * the arguments a tool actually received (extensions/rbtr/args.ts).
 */

import { describe, expect, test } from "vitest";

import { decodeStringList, echoArgs } from "../extensions/rbtr/args.js";

describe("decodeStringList", () => {
  test("passes native arrays through", () => {
    expect(decodeStringList(["a.py", "b.py"])).toEqual(["a.py", "b.py"]);
    expect(decodeStringList(["a.py"])).toEqual(["a.py"]);
  });

  test("decodes a bare JSON-encoded array string", () => {
    expect(decodeStringList('["a.py", "b.py"]')).toEqual(["a.py", "b.py"]);
  });

  test("unwraps a one-element list holding a JSON-encoded array", () => {
    expect(decodeStringList(['["a.py"]'])).toEqual(["a.py"]);
  });

  test("keeps a genuine single value that is not JSON", () => {
    expect(decodeStringList(["src/a.py"])).toEqual(["src/a.py"]);
    expect(decodeStringList("main")).toEqual(["main"]);
  });

  test("returns empty for missing or non-list values", () => {
    expect(decodeStringList(undefined)).toEqual([]);
    expect(decodeStringList(42)).toEqual([]);
  });
});

describe("echoArgs", () => {
  test("is empty when none of the keys are present", () => {
    expect(echoArgs({ symbol: "x" }, ["file_paths"])).toBe("");
    expect(echoArgs({}, ["query", "keywords"])).toBe("");
  });

  test("echoes present args verbatim as JSON, so malformations are visible", () => {
    // A double-encoded file_paths surfaces exactly as received.
    expect(echoArgs({ file_paths: ['["src/a.py"]'] }, ["file_paths"])).toBe(
      '\n\nArguments received: file_paths=["[\\"src/a.py\\"]"]',
    );
  });

  test("includes only the named keys that are defined", () => {
    expect(echoArgs({ query: "retry", scope: undefined, keywords: ["a"] }, ["query", "scope", "keywords"])).toBe(
      '\n\nArguments received: query="retry", keywords=["a"]',
    );
  });
});
