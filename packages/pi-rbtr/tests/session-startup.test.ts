/**
 * A pi session's lifecycle: the tools it starts with.
 */

import { expect } from "vitest";

import { ON_REQUEST } from "../extensions/rbtr/tool-set.js";
import { test } from "./session.fixtures.js";

test("the housekeeping tools start inactive, and rbtr_index_tools makes them active", async ({ pi }) => {
  await pi.start();
  const active = () => pi.runtime.session.getActiveToolNames();
  expect(active()).toEqual(
    expect.arrayContaining([
      "rbtr_search",
      "rbtr_read_symbol",
      "rbtr_find_refs",
      "rbtr_changed_symbols",
      "rbtr_list_symbols",
      "rbtr_watch",
      "rbtr_index_tools",
    ]),
  );
  expect(active().filter((name) => ON_REQUEST.includes(name))).toEqual([]);
  await pi.run({ name: "rbtr_index_tools", args: {} });
  expect(active()).toEqual(expect.arrayContaining([...ON_REQUEST]));
});
