/**
 * What the model gets back from each rbtr tool, called by the model in a
 * pi session over the run's indexed repository, and what pi is told about
 * the tools.
 */

import type { TSchema } from "typebox";
import { Check } from "typebox/schema";
import { describe, expect, inject } from "vitest";

import type { Request } from "../extensions/rbtr/generated/protocol.js";
import { type Ask, type Call, type PiSession, type ToolEnd, test, textOf } from "./session.fixtures.js";

/** The output schema *name* declares in the session. */
function outputSchema(pi: PiSession, name: string): TSchema {
  const schema = pi.runtime.session.getToolDefinition(name)?.outputSchema;
  if (schema === undefined) throw new Error(`${name} declares no output schema`);
  return schema;
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
      expect(reply, label).toEqual(await ask({ ...expected.ask, repo_path: cwd } as Request));
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
});

describe("with a daemon of its own", () => {
  test.override({ ownDaemon: true });

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
