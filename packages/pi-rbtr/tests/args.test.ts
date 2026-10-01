/**
 * Tests for the argument helpers in extensions/rbtr/args.ts.
 */

import { describe, expect, test } from "vitest";

import { commandRefs, decodeStringList } from "../extensions/rbtr/args.js";

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

describe("commandRefs", () => {
  test("takes each whitespace-separated word as a ref", () => {
    expect(commandRefs("main  feature-x\t v1.2")).toEqual(["main", "feature-x", "v1.2"]);
  });

  test("gives no refs for empty arguments, so the default applies", () => {
    expect(commandRefs("")).toEqual([]);
    expect(commandRefs("   ")).toEqual([]);
  });
});
