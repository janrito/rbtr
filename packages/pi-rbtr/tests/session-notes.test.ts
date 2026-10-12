/**
 * How pi-rbtr's hook adds the `[rbtr index]` note to pi's own `bash` and
 * `read` results, in a session over the run's indexed repository: which
 * results get one, and that it holds the facts the index gives for the
 * result. What the facts are is tested in `annotate.test.ts`.
 */

import { readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fauxAssistantMessage, fauxText } from "@earendil-works/pi-ai";
import { describe, expect, inject } from "vitest";

import { HEADER, type IndexLookups, readFacts, searchFacts } from "../extensions/rbtr/annotate.js";
import { send } from "../extensions/rbtr/daemon-client.js";
import { type ToolEnd, test, textOf } from "./session.fixtures.js";

const { subject, long } = inject("corpus");

/** The lines of the note on *end*'s result, or null when it has none. */
function note(end: ToolEnd): string[] | null {
  const text = textOf(end.result);
  const at = text.indexOf(HEADER);
  return at === -1
    ? null
    : text
        .slice(at + HEADER.length)
        .trim()
        .split("\n");
}

/** The lookups the hook makes (`index.ts`), asked of the daemon at *rpc* about *repo*. */
function lookups(rpc: string, repo: string): IndexLookups {
  const daemon = { rpcEndpoint: rpc };
  return {
    readSymbol: (symbol) => send({ kind: "read_symbol", repo_path: repo, symbol }, daemon),
    findRefs: (symbol) => send({ kind: "find_refs", repo_path: repo, symbol }, daemon),
    search: (query) => send({ kind: "search", repo_path: repo, query, limit: 3 }, daemon),
    listSymbols: (file_path) => send({ kind: "list_symbols", repo_path: repo, file_path }, daemon),
  };
}

const searchSubject = { name: "bash", args: { command: `grep -rln ${subject.name} ${subject.user}` } };

test("bash: a search in the repository gets the facts the index gives for it", async ({
  shared,
  repo,
  piReady: pi,
}) => {
  const [end] = await pi.run(searchSubject);
  const output = textOf({ ...end.result, content: end.result.content.slice(0, 1) });
  const facts = await searchFacts(lookups(shared.rpc, repo), searchSubject.args.command, output);
  expect(facts).not.toEqual([]);
  expect(note(end)).toEqual(facts);
});

test("bash: a failed search gets none", async ({ piReady: pi }) => {
  const [end] = await pi.run({ name: "bash", args: { command: `grep -rn ${subject.name} missing-dir` } });
  expect(end.isError).toBe(true);
  expect(note(end)).toBeNull();
});

test("bash: the note follows the output, and the structured result is kept", async ({ piReady: pi }) => {
  const [end] = await pi.run(searchSubject);
  expect(end.result.content[0]).toEqual({ type: "text", text: `${subject.user}\n` });
  expect(end.result.structuredContent).toMatchObject({ output: `${subject.user}\n`, exit_code: 0 });
  expect(note(end)).not.toBeNull();
});

test("bash: a search outside the repository gets none", async ({ outside, piReady: pi }) => {
  const [end] = await pi.run({ name: "bash", args: { command: `cd ${outside} && grep -rln ${subject.name} .` } });
  expect(end.isError).toBe(false);
  expect(note(end)).toBeNull();
});

describe("with a daemon of its own", () => {
  test.override({ ownDaemon: true });

  test("bash: with the daemon gone, a search gets none and its output is kept", async ({ cli, piReady: pi }) => {
    cli("daemon", "stop");
    const [end] = await pi.run(searchSubject);
    expect(end.result.content).toEqual([{ type: "text", text: `${subject.user}\n` }]);
  });
});

const readLong = { name: "read", args: { path: long } };

test("read: a long file's first read gets the outline the index gives for it", async ({
  shared,
  repo,
  piReady: pi,
}) => {
  const [end] = await pi.run(readLong);
  const lineCount = readFileSync(join(repo, long), "utf8").split("\n").length;
  const facts = await readFacts(lookups(shared.rpc, repo), long, lineCount);
  expect(facts).not.toEqual([]);
  expect(note(end)).toEqual(facts);
});

test.for<[string, { path: string; offset?: number }]>([
  ["a file shorter than 200 lines", { path: subject.file }],
  ["part of a long file", { path: long, offset: 100 }],
])("read: %s gets none", async ([, args], { piReady: pi }) => {
  const [end] = await pi.run({ name: "read", args });
  expect(note(end)).toBeNull();
});

test("read: two reads of a long file in one turn get one outline", async ({ piReady: pi }) => {
  const ends = await pi.run([readLong, readLong]);
  expect(ends.map((end) => note(end) !== null).sort()).toEqual([false, true]);
});

test("read: a file outlined in the context is not outlined again, until the outline leaves it", async ({
  piReady: pi,
}) => {
  const [first] = await pi.run(readLong);
  expect(note(first)).not.toBeNull();
  const [again] = await pi.run(readLong);
  expect(note(again)).toBeNull();
  // Back to before the first read: its outline is no longer in the context.
  const firstPrompt = pi.runtime.session.sessionManager
    .getEntries()
    .find((entry) => entry.type === "message" && entry.message.role === "user");
  if (firstPrompt === undefined) throw new Error("the session has no prompt");
  await pi.runtime.session.navigateTree(firstPrompt.id);
  const [afterNavigating] = await pi.run(readLong);
  expect(note(afterNavigating)).not.toBeNull();
});

describe("with pi keeping little when it compacts", () => {
  test.override({ piSettings: { compaction: { keepRecentTokens: 1 } } });

  test("read: an outline compaction summarises away is given again", async ({ piReady: pi }) => {
    const [first] = await pi.run(readLong);
    expect(note(first)).not.toBeNull();
    // A second turn, so compaction has an earlier one to summarise.
    await pi.run();
    // Compaction asks the model for a summary of the history and one of the turn it splits.
    pi.faux.setResponses([fauxAssistantMessage(fauxText("summary")), fauxAssistantMessage(fauxText("summary"))]);
    await pi.runtime.session.compact();
    const [afterCompacting] = await pi.run(readLong);
    expect(note(afterCompacting)).not.toBeNull();
  });
});

describe("in a subdirectory of the repository", () => {
  test.override("cwd", ({ repo }) => join(repo, dirname(long)));

  test("read: a long file's outline names it from the repository's root", async ({ piReady: pi }) => {
    const [end] = await pi.run({ name: "read", args: { path: basename(long) } });
    expect(note(end)?.[0]).toBe(`outline of ${long}:`);
  });
});

describe("with codemode", () => {
  test.override({ piSettings: { defaultTools: ["+codemode"] } });

  // A script's call returns to the script, not to the model. The script
  // gets `read`'s structured result, which a note leaves as it is, so the
  // test reads the result pi reports for the call (`tool_execution_end`).
  test("read: a file a script reads gets none", async ({ piReady: pi }) => {
    const nested: string[] = [];
    const stop = pi.runtime.session.subscribe((event) => {
      if (event.type === "tool_execution_end" && event.parentToolCallId !== undefined) {
        nested.push(textOf(event.result));
      }
    });
    const [script] = await pi.run({
      name: "codemode",
      args: { code: `await tools.read({ path: ${JSON.stringify(long)} });` },
    });
    stop();
    expect(script.isError).toBe(false);
    expect(nested).toHaveLength(1);
    expect(nested[0]).not.toContain(HEADER);
  });
});
