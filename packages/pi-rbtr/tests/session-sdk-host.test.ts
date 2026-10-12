/**
 * pi-rbtr in a session whose SDK host passes no UI and never initialises
 * pi's theme, as pi's SDK examples do. Its own file: any session with a UI
 * initialises the theme for the whole file.
 */

import { expect, inject } from "vitest";

import { test, textOf } from "./session.fixtures.js";

const { subject } = inject("corpus");

test("a host with no UI gets the daemon's answers", async ({ ask, repo, sdkHost }) => {
  await sdkHost.start();
  const [end] = await sdkHost.run({ name: "rbtr_read_symbol", args: { symbol: subject.name } });
  expect(end.result.structuredContent).toEqual(
    await ask({ kind: "read_symbol", repo_path: repo, symbol: subject.name }),
  );
  expect(end.isError).toBe(false);
});

test("a host with no UI cannot delete index data", async ({ sdkHost }) => {
  await sdkHost.start();
  const [, gc] = await sdkHost.run(
    { name: "rbtr_index_tools", args: {} },
    { name: "rbtr_gc", args: { dry_run: false } },
  );
  expect(gc.isError).toBe(true);
  expect(textOf(gc.result)).toContain("run `rbtr gc` in a terminal");
});
