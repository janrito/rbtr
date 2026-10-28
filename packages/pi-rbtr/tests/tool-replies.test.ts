/**
 * What the model gets back from each rbtr tool, called by the model in a
 * pi session over the run's indexed repository, and what pi is told about
 * the tools.
 */

import type { ToolAnnotations } from "@earendil-works/pi-coding-agent";
import type { TSchema } from "typebox";
import { Check } from "typebox/schema";
import { describe, expect, inject } from "vitest";

import type { Request } from "../extensions/rbtr/generated/protocol.js";
import { type Ask, type Call, type PiSession, type ToolEnd, test, textOf, until } from "./session.fixtures.js";

/** The output schema *name* declares in the session. */
function outputSchema(pi: PiSession, name: string): TSchema {
  const schema = pi.runtime.session.getToolDefinition(name)?.outputSchema;
  if (schema === undefined) throw new Error(`${name} declares no output schema`);
  return schema;
}

/** The rule pi's docs give for the calls Codex asks approval for (docs/extensions.md, "Tool exposure"). */
function needsApproval(hints: ToolAnnotations | undefined): boolean {
  return (
    hints?.destructiveHint === true ||
    (!hints?.readOnlyHint && ((hints?.destructiveHint ?? true) || (hints?.openWorldHint ?? true)))
  );
}

/** A reply rbtr writes; one of kind `error` comes flagged as an error. */
type Reply = { kind: string } & Record<string, unknown>;

/**
 * What a call's reply must be: the daemon's reply to a request, asked about
 * the session's repository, which shows what `shows` holds; or a reply the
 * extension writes itself.
 */
type Expected = { ask: DistributiveOmit<Request, "repo_path">; shows: Reply } | { reply: Reply };

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

const { subject } = inject("corpus");

/** A name nothing defines. */
const unknown = `${subject.name}_defined_nowhere`;

/** A call at the ref no daemon has indexed, which every read refuses. */
const unindexed = { ref: "unindexed" };
const refused = { kind: "error", code: "index_not_built" };

/** Run each step's call in one session over *cwd*, and check each reply against its expectation. */
async function checkReplies(pi: PiSession, ask: Ask, cwd: string, steps: [Call, Expected][]): Promise<ToolEnd[]> {
  const ends = await pi.run(...steps.map(([call]) => call));
  expect(ends.map((end) => end.name)).toEqual(steps.map(([call]) => call.name));
  for (const [i, end] of ends.entries()) {
    const expected = steps[i][1];
    const reply = end.result.structuredContent as Reply;
    const label = `${end.name} ${JSON.stringify(end.input)}`;
    if ("ask" in expected) {
      // DuckDB sums a lexical score's terms in parallel, in no fixed order, so two
      // identical searches can score a hit differently in the last digits.
      const rounded = (value: unknown): unknown =>
        JSON.parse(
          JSON.stringify(value, (_key, field) =>
            typeof field === "number" && !Number.isInteger(field) ? Number(field.toPrecision(12)) : field,
          ),
        );
      expect(rounded(reply), label).toEqual(rounded(await ask({ ...expected.ask, repo_path: cwd } as Request)));
      expect(reply, label).toMatchObject(expected.shows);
    } else {
      expect(reply, label).toMatchObject(expected.reply);
    }
    expect(end.isError, label).toBe(reply.kind === "error");
    expect(JSON.parse(textOf(end.result)), label).toEqual(reply);
    const schema = outputSchema(pi, end.name);
    expect(Check(schema, reply), label).toBe(true);
    // The schema is the tool's own: a reply of another kind fails it.
    expect(Check(schema, { kind: "not_a_reply" }), label).toBe(false);
  }
  return ends;
}

// Each row is one session, in which the model calls one tool, in order.
test.for<[string, [Call, Expected][]]>([
  [
    "rbtr_read_symbol",
    [
      [
        { name: "rbtr_read_symbol", args: { symbol: subject.name } },
        {
          ask: { kind: "read_symbol", symbol: subject.name },
          shows: {
            kind: "read_symbol",
            chunks: expect.arrayContaining([expect.objectContaining({ file_path: subject.file })]),
          },
        },
      ],
      [
        { name: "rbtr_read_symbol", args: { symbol: unknown } },
        { ask: { kind: "read_symbol", symbol: unknown }, shows: { kind: "read_symbol", chunks: [] } },
      ],
      [
        { name: "rbtr_read_symbol", args: { symbol: subject.name, ...unindexed } },
        { ask: { kind: "read_symbol", symbol: subject.name, ...unindexed }, shows: refused },
      ],
    ],
  ],
  [
    "rbtr_find_refs",
    [
      [
        { name: "rbtr_find_refs", args: { symbol: subject.name } },
        {
          ask: { kind: "find_refs", symbol: subject.name },
          shows: {
            kind: "find_refs",
            refs: expect.arrayContaining([expect.objectContaining({ file_path: subject.user })]),
          },
        },
      ],
      [
        { name: "rbtr_find_refs", args: { symbol: unknown } },
        { ask: { kind: "find_refs", symbol: unknown }, shows: { kind: "find_refs", refs: [] } },
      ],
      [
        { name: "rbtr_find_refs", args: { symbol: subject.name, ...unindexed } },
        { ask: { kind: "find_refs", symbol: subject.name, ...unindexed }, shows: refused },
      ],
    ],
  ],
  [
    "rbtr_list_symbols",
    [
      [
        { name: "rbtr_list_symbols", args: { file: subject.file } },
        {
          ask: { kind: "list_symbols", file_path: subject.file },
          shows: {
            kind: "list_symbols",
            chunks: expect.arrayContaining([expect.objectContaining({ name: subject.name })]),
          },
        },
      ],
      [
        { name: "rbtr_list_symbols", args: { file: `${subject.file}.missing` } },
        {
          ask: { kind: "list_symbols", file_path: `${subject.file}.missing` },
          shows: { kind: "list_symbols", chunks: [] },
        },
      ],
      [
        { name: "rbtr_list_symbols", args: { file: subject.file, ...unindexed } },
        { ask: { kind: "list_symbols", file_path: subject.file, ...unindexed }, shows: refused },
      ],
    ],
  ],
  [
    "rbtr_changed_symbols",
    [
      [
        { name: "rbtr_changed_symbols", args: { base: "base", head: "HEAD" } },
        {
          ask: { kind: "changed_symbols", base: "base", head: "HEAD" },
          shows: {
            kind: "changed_symbols",
            changes: expect.arrayContaining([
              expect.objectContaining({ chunk: expect.objectContaining({ name: subject.name }) }),
            ]),
          },
        },
      ],
      [
        { name: "rbtr_changed_symbols", args: { base: "HEAD", head: "HEAD" } },
        {
          ask: { kind: "changed_symbols", base: "HEAD", head: "HEAD" },
          shows: { kind: "changed_symbols", changes: [] },
        },
      ],
      [
        { name: "rbtr_changed_symbols", args: { base: "unindexed", head: "HEAD" } },
        { ask: { kind: "changed_symbols", base: "unindexed", head: "HEAD" }, shows: refused },
      ],
    ],
  ],
  [
    "rbtr_search",
    [
      [
        { name: "rbtr_search", args: { query: subject.name } },
        {
          ask: { kind: "search", query: subject.name },
          shows: { kind: "search", results: expect.arrayContaining([expect.objectContaining({ name: subject.name })]) },
        },
      ],
      [
        { name: "rbtr_search", args: { query: subject.name, ...unindexed } },
        { ask: { kind: "search", query: subject.name, ...unindexed }, shows: refused },
      ],
    ],
  ],
  [
    "rbtr_index_tools, then the tools it loads",
    [
      [{ name: "rbtr_index_tools", args: {} }, { reply: { kind: "tools_loaded", tools: ["rbtr_status", "rbtr_gc"] } }],
      // A status and what a collection would delete name the daemon's work
      // on every repository: other test files change them meanwhile.
      [
        { name: "rbtr_status", args: {} },
        {
          reply: {
            kind: "status",
            indexed_refs: expect.arrayContaining([
              expect.objectContaining({ names: expect.arrayContaining(["base"]) }),
            ]),
          },
        },
      ],
      [{ name: "rbtr_gc", args: { dry_run: true } }, { reply: { kind: "gc", dry_run: true } }],
    ],
  ],
])(
  "%s: each reply is rbtr's response, the same to the model and to scripts",
  async ([, steps], { ask, repo, processes, pi }) => {
    await pi.start();
    await checkReplies(pi, ask, repo, steps);
    // The daemon gave every reply, refusals included: no rbtr process ran but startup's.
    expect(processes()).toEqual(["python3 -m rbtr --json daemon status"]);
  },
);

describe("in a repository of the test's own", () => {
  test.override({ projectSettings: { autoIndex: false } });
  test.override("cwd", ({ writableRepo }) => writableRepo);

  // Watching changes what the daemon holds, so these replies are rbtr's own words.
  test("rbtr_watch: each reply is rbtr's response, the same to the model and to scripts", async ({
    ask,
    writableRepo,
    pi,
  }) => {
    await pi.start();
    await checkReplies(pi, ask, writableRepo, [
      [
        { name: "rbtr_watch", args: { refs: ["scratch"] } },
        {
          reply: { kind: "watch_set", watched: expect.arrayContaining([expect.objectContaining({ ref: "scratch" })]) },
        },
      ],
      [
        { name: "rbtr_watch", args: { refs: ["scratch"], remove: true } },
        { reply: { kind: "unwatch", dry_run: false } },
      ],
      [{ name: "rbtr_watch", args: { remove_stale: true } }, { reply: { kind: "unwatch", removed: {} } }],
      [
        { name: "rbtr_watch", args: { refs: ["HEAD"], remove: true } },
        { reply: { kind: "error", message: "HEAD cannot be removed from the watch set" } },
      ],
    ]);
  });

  describe("while it builds", () => {
    test.override({ longBuild: true });

    // A read of a ref the daemon is building waits for it, then answers.
    test("a read at a ref being built shows the build's progress, then answers", async ({ pi }) => {
      await pi.start();
      const [, read] = await pi.run(
        { name: "rbtr_watch", args: { refs: ["HEAD"] } },
        { name: "rbtr_read_symbol", args: { symbol: subject.name } },
      );
      expect(read.isError, textOf(read.result)).toBe(false);
      expect(read.result.structuredContent).toMatchObject({
        kind: "read_symbol",
        chunks: expect.arrayContaining([expect.objectContaining({ name: subject.name })]),
      });
      expect(read.updates.length).toBeGreaterThan(0);
      expect(textOf(read.updates[0])).toMatch(/^waiting for the build/);
    });

    test("an abort while a read waits for a build ends it at once", async ({ pi }) => {
      await pi.start();
      let waiting = false;
      const unsubscribe = pi.runtime.session.subscribe((event) => {
        if (event.type === "tool_execution_update") waiting = true;
      });
      const run = pi.run(
        { name: "rbtr_watch", args: { refs: ["HEAD"] } },
        { name: "rbtr_read_symbol", args: { symbol: subject.name } },
      );
      await until(() => waiting);
      unsubscribe();
      const started = Date.now();
      await pi.runtime.session.abort();
      const [, read] = await run;
      expect(Date.now() - started).toBeLessThan(1_000);
      expect(read.isError).toBe(true);
    });
  });
});

test("a list argument reaches the tool as the list however the model encoded it, and one left out stays left out", async ({
  pi,
}) => {
  const encodings: [string, (list: string[]) => string | string[]][] = [
    ["JSON text", (list) => JSON.stringify(list)],
    ["a list holding JSON text", (list) => [JSON.stringify(list)]],
    ["a list", (list) => list],
  ];
  const lists: [string, string, string[], Record<string, string>][] = [
    ["rbtr_search", "keywords", ["config", "settings"], { query: subject.name }],
    ["rbtr_search", "variants", ["read the settings"], { query: subject.name }],
    ["rbtr_read_symbol", "file_paths", [subject.file], { symbol: subject.name }],
    ["rbtr_find_refs", "file_paths", [subject.user], { symbol: subject.name }],
    ["rbtr_changed_symbols", "file_paths", [subject.file], { base: "HEAD", head: "HEAD" }],
    // Refs the shared daemon watches already: watching them changes nothing.
    ["rbtr_watch", "refs", ["HEAD", "base"], {}],
  ];
  await pi.start();
  for (const [encoding, encode] of encodings) {
    const ends = await pi.run(
      ...lists.map(([name, key, list, rest]): Call => ({ name, args: { ...rest, [key]: encode(list) } })),
    );
    expect(
      ends.map((end) => [end.name, end.isError, end.input]),
      encoding,
    ).toEqual(lists.map(([name, key, list, rest]) => [name, false, { ...rest, [key]: list }]));
  }
  const [leftOut] = await pi.run({ name: "rbtr_read_symbol", args: { symbol: subject.name } });
  expect(leftOut.input).toEqual({ symbol: subject.name });
});

describe("with a daemon of its own", () => {
  test.override({ ownDaemon: true });

  // Deleting asks the user, whatever the model passed.
  const deleting: Call[] = [
    { name: "rbtr_index_tools", args: {} },
    { name: "rbtr_gc", args: { dry_run: false } },
  ];

  test("rbtr_gc deletes nothing the user declines", async ({ pi }) => {
    await pi.start();
    pi.ui.confirmAnswers.push(false);
    const [, gc] = await pi.run(...deleting);
    expect(gc.isError).toBe(true);
    expect(textOf(gc.result)).toContain("declined");
  });

  test("rbtr_gc deletes what the user accepts", async ({ pi }) => {
    await pi.start();
    pi.ui.confirmAnswers.push(true);
    const [, gc] = await pi.run(...deleting);
    expect(gc.isError).toBe(false);
    expect(gc.result.structuredContent).toMatchObject({ kind: "gc", dry_run: false });
  });

  test("with the daemon gone, a found and a refused reply come from the CLI and are the same", async ({
    cli,
    processes,
    pi,
  }) => {
    const calls: Call[] = [
      { name: "rbtr_read_symbol", args: { symbol: subject.name } },
      { name: "rbtr_read_symbol", args: { symbol: subject.name, ...unindexed } },
    ];
    await pi.start();
    const fromDaemon = await pi.run(...calls);
    cli("daemon", "stop");
    const fromCli = await pi.run(...calls);
    expect(processes()).toEqual(
      expect.arrayContaining([
        `python3 -m rbtr --json read-symbol ${subject.name}`,
        `python3 -m rbtr --json read-symbol ${subject.name} --ref unindexed`,
      ]),
    );
    expect(fromDaemon.map((end) => end.isError)).toEqual([false, true]);
    expect(fromCli.map((end) => [end.isError, end.result.structuredContent])).toEqual(
      fromDaemon.map((end) => [end.isError, end.result.structuredContent]),
    );
    for (const end of fromCli) expect(JSON.parse(textOf(end.result))).toEqual(end.result.structuredContent);
  });
});

describe("with codemode", () => {
  test.override({ piSettings: { defaultTools: ["+codemode"] } });

  // A tool that asks the user, or loads tools, is the model's to call.
  test("a script can call the loaded rbtr_status, but not rbtr_gc or the loader", async ({ pi }) => {
    await pi.start();
    await pi.run({ name: "rbtr_index_tools", args: {} });
    // `tools` throws on a name a script cannot call.
    const code = `const callable = (name) => { try { return typeof tools[name] === "function"; } catch { return false; } };
text(["rbtr_status", "rbtr_gc", "rbtr_index_tools"].map(callable).join(" "));`;
    const [script] = await pi.run({ name: "codemode", args: { code } });
    expect(textOf(script.result)).toContain("true false false");
  });
});

test("pi lists the rbtr tools in one namespace, with the hints a permission extension reads", async ({ pi }) => {
  await pi.start();
  const tools = pi.runtime.session.getAllTools().filter((tool) => tool.name.startsWith("rbtr_"));
  expect(tools.map((tool) => [tool.name, tool.namespace?.name])).toEqual(tools.map((tool) => [tool.name, "rbtr"]));
  expect(tools.filter((tool) => tool.annotations?.openWorldHint !== false).map((tool) => tool.name)).toEqual([]);
  expect(tools.filter((tool) => needsApproval(tool.annotations)).map((tool) => tool.name)).toEqual(["rbtr_gc"]);
  expect(tools.find((tool) => tool.name === "rbtr_watch")?.annotations).toEqual({
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  });
});

test("the system prompt lists the tools that find code first, with their rules", async ({ pi }) => {
  const finding = ["rbtr_search", "rbtr_read_symbol", "rbtr_find_refs", "rbtr_changed_symbols", "rbtr_list_symbols"];
  await pi.start();
  await pi.run();
  const prompt = pi.runtime.session.systemPrompt;
  const tools = pi.runtime.session.getAllTools().filter((tool) => tool.name.startsWith("rbtr_"));
  const listed = tools
    .map((tool) => tool.name)
    .filter((name) => prompt.includes(name))
    .sort((a, b) => prompt.indexOf(a) - prompt.indexOf(b));
  expect(listed.slice(0, finding.length)).toEqual(finding);
  // pi adds the rules of active tools only.
  const active = new Set(pi.runtime.session.getActiveToolNames());
  const shown = tools.filter((tool) => active.has(tool.name)).flatMap((tool) => tool.promptGuidelines ?? []);
  expect(shown.filter((rule) => !prompt.includes(rule))).toEqual([]);
  // Every rule, once the housekeeping tools are loaded: few enough to be read.
  expect(tools.flatMap((tool) => tool.promptGuidelines ?? []).length).toBeLessThanOrEqual(16);
  for (const tool of tools.filter((tool) => finding.includes(tool.name))) {
    expect(tool.description, tool.name).toMatch(/instead of/);
  }
});
