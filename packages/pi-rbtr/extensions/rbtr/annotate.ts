/**
 * Index facts for the output of the tools the agent already uses.
 *
 * Agents keep to `bash` and `read` whatever other tools they are given,
 * so the index's facts are appended to those tools' output: after a
 * search for a symbol, where it is defined and what uses it; after a
 * search for a phrase, the index's best matches; after the first read of
 * a large file, its outline. At most ten lines, headed `[rbtr index]`,
 * and nothing when the index has nothing to add.
 */

import { homedir } from "node:os";
import { resolve, sep } from "node:path";

import type { SessionEntry } from "@earendil-works/pi-coding-agent";

import type {
  ChunkKind,
  FindRefsResponse,
  ListSymbolsResponse,
  ReadSymbolResponse,
  SearchResponse,
} from "./generated/protocol.js";

export const HEADER = "[rbtr index]";
const MAX_LINES = 10;
/** A read of a file shorter than this gets no outline: reading it whole is cheap. */
export const OUTLINE_FROM_LINES = 200;
/** An outline name longer than this is cut, so each line stays one short line. */
const MAX_NAME_CHARS = 80;
/**
 * Chunks that name something without defining code. A search for a name
 * shows them only when no code chunk defines it.
 */
const NOT_CODE: ReadonlySet<ChunkKind> = new Set(["doc_section", "config_key", "comment", "raw_chunk", "import"]);
/** Chunks an outline leaves out: they are not where a reader goes next. */
const NOT_OUTLINED: ReadonlySet<ChunkKind> = new Set(["import", "comment"]);

/** The lookups the facts come from, against the repository the agent works in. */
export interface IndexLookups {
  readSymbol(symbol: string): Promise<ReadSymbolResponse>;
  findRefs(symbol: string): Promise<FindRefsResponse>;
  search(query: string): Promise<SearchResponse>;
  listSymbols(filePath: string): Promise<ListSymbolsResponse>;
}

const OPERATORS = new Set(["|", "||", "&&", ";", "&"]);

/**
 * A command's shell words, quotes removed, with `|`, `||`, `&&`, `;` and
 * `&` as words of their own. Enough of the shell to find a search's
 * pattern; not a parser for anything else.
 */
export function shellWords(command: string): string[] {
  const words: string[] = [];
  let word = "";
  let quote: "'" | '"' | null = null;
  let inWord = false;
  const flush = () => {
    if (inWord) words.push(word);
    word = "";
    inWord = false;
  };
  for (let i = 0; i < command.length; i++) {
    const c = command[i];
    if (quote) {
      if (c === quote) quote = null;
      else if (c === "\\" && quote === '"' && i + 1 < command.length) word += command[++i];
      else word += c;
      continue;
    }
    if (c === "'" || c === '"') {
      quote = c;
      inWord = true;
    } else if (c === "\\" && i + 1 < command.length) {
      word += command[++i];
      inWord = true;
    } else if (/\s/.test(c)) {
      flush();
    } else if (c === "|" || c === "&" || c === ";") {
      flush();
      const pair = command.slice(i, i + 2);
      if (pair === "||" || pair === "&&") {
        words.push(pair);
        i++;
      } else {
        words.push(c);
      }
    } else {
      word += c;
      inWord = true;
    }
  }
  flush();
  return words;
}

/** Search options that take the next word as their value. */
const TAKES_VALUE = new Set([
  "-A",
  "-B",
  "-C",
  "-m",
  "-f",
  "-g",
  "-t",
  "-T",
  "-d",
  "--glob",
  "--type",
  "--type-not",
  "--max-count",
  "--max-depth",
  "--include",
  "--exclude",
  "--context",
  "--after-context",
  "--before-context",
]);

const SEARCH_COMMANDS = new Set(["grep", "egrep", "fgrep", "rg", "ag"]);

/**
 * A search's pattern and the paths it searches, from its arguments: the
 * first `-e`/`--regexp` value, every operand then being a path; or else
 * the first operand.
 */
function searchArgs(args: string[]): { pattern: string | null; paths: string[] } {
  let given: string | null = null;
  const operands: string[] = [];
  let options = true;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (options && arg === "--") {
      options = false;
    } else if (options && (arg === "-e" || arg === "--regexp")) {
      given ??= args[i + 1] ?? null;
      i++;
    } else if (options && arg.startsWith("--regexp=")) {
      given ??= arg.slice("--regexp=".length);
    } else if (options && arg.startsWith("-")) {
      if (TAKES_VALUE.has(arg)) i++;
    } else {
      operands.push(arg);
    }
  }
  if (given !== null) return { pattern: given, paths: operands };
  const [pattern = null, ...paths] = operands;
  return { pattern, paths };
}

interface Search {
  pattern: string | null;
  paths: string[];
  /** The directories `cd` moved to before the search, in order. */
  cds: string[];
}

/** The first `grep`, `rg`, `ag` or `git grep` in a command, if any. */
function findSearch(command: string): Search | null {
  const words = shellWords(command);
  let stage: string[] = [];
  const stages: string[][] = [];
  for (const word of words) {
    if (OPERATORS.has(word)) {
      stages.push(stage);
      stage = [];
    } else {
      stage.push(word);
    }
  }
  stages.push(stage);
  const cds: string[] = [];
  for (const [head, second, ...rest] of stages) {
    if (head === "cd") cds.push(second ?? "~");
    if (head === "git" && second === "grep") return { ...searchArgs(rest), cds };
    if (head !== undefined && SEARCH_COMMANDS.has(head))
      return { ...searchArgs([second, ...rest].filter((w) => w !== undefined)), cds };
  }
  return null;
}

/** The pattern of the first `grep`, `rg`, `ag` or `git grep` in a command, if any. */
export function searchPattern(command: string): string | null {
  return findSearch(command)?.pattern ?? null;
}

/** *path* resolved against *dir*, with a leading `~` as the home directory. */
function resolveFrom(dir: string, path: string): string {
  if (path === "~" || path.startsWith("~/")) return resolve(homedir(), path.slice(2));
  return resolve(dir, path);
}

/**
 * Whether a command's search looks only outside *root*: every path it
 * names, or the directory it runs in when it names none, after the
 * `cd`s before it, starting from *cwd*. A command with no search does not.
 */
export function searchesOutside(command: string, cwd: string, root: string): boolean {
  const search = findSearch(command);
  if (search === null) return false;
  const dir = search.cds.reduce(resolveFrom, cwd);
  const targets = search.paths.length > 0 ? search.paths.map((path) => resolveFrom(dir, path)) : [dir];
  return targets.every((target) => target !== root && !target.startsWith(root + sep));
}

const DEFINITION = /^(?:def|class|function|fn|func|struct|interface|type)\s+([A-Za-z_]\w*)/;
const NAME = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/;

/**
 * The symbol a search pattern names, if it names one: `QuerySet.union`,
 * `\bunion\b`, or a definition such as `def union`.
 */
export function searchedName(pattern: string): string | null {
  const bare = pattern
    .replace(/\\b|\\</g, "")
    .replace(/\\>/g, "")
    .trim();
  const name = DEFINITION.exec(bare)?.[1] ?? bare;
  return NAME.test(name) && name.length >= 3 ? name : null;
}

/** The words of a search pattern that is a phrase rather than a name, if it is one. */
export function searchedPhrase(pattern: string): string | null {
  const words = pattern
    .replace(/\\[a-zA-Z]/g, " ")
    .replace(/[\\^$.*+?()[\]{}|]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 0);
  return words.length >= 2 ? words.join(" ") : null;
}

/** Facts for a `bash` search's output: where its symbol is defined and used, or its phrase's matches. */
export async function searchFacts(index: IndexLookups, command: string, output: string): Promise<string[]> {
  const pattern = searchPattern(command);
  if (pattern === null) return [];
  const name = searchedName(pattern);
  if (name !== null) {
    const chunks = (await index.readSymbol(name)).chunks;
    const code = chunks.filter((c) => !NOT_CODE.has(c.kind));
    const definitions = (code.length > 0 ? code : chunks).slice(0, 3);
    if (definitions.length === 0) return [];
    const lines = definitions
      .filter((d) => !output.includes(`${d.file_path}:${d.line_start}`))
      .map((d) => `defined: ${d.file_path}:${d.line_start} ${d.kind} ${d.name}`);
    const users = [...new Set((await index.findRefs(name)).refs.map((r) => r.file_path))];
    if (users.length > 0) {
      const more = users.length > 5 ? ` (+${users.length - 5} more)` : "";
      lines.push(`used by: ${users.slice(0, 5).join(", ")}${more}`);
    }
    return lines;
  }
  const phrase = searchedPhrase(pattern);
  if (phrase === null) return [];
  return (await index.search(phrase)).results
    .slice(0, 3)
    .map((h) => `match: ${h.file_paths[0]}:${h.line_start} ${h.kind} ${h.name}`);
}

/** Facts for the first read of a file: its outline, when it is long enough to be worth one. */
export async function readFacts(index: IndexLookups, filePath: string, lineCount: number): Promise<string[]> {
  if (lineCount < OUTLINE_FROM_LINES) return [];
  const symbols = (await index.listSymbols(filePath)).chunks.filter((s) => s.name !== "" && !NOT_OUTLINED.has(s.kind));
  if (symbols.length === 0) return [];
  const shown = symbols
    .slice(0, MAX_LINES - 2)
    .map((s) => `  ${cutName(s.name)} (${s.kind}) lines ${s.line_start}-${s.line_end}`);
  const more = symbols.length > shown.length ? [`  +${symbols.length - shown.length} more: rbtr_list_symbols`] : [];
  return [outlineHeading(filePath), ...shown, ...more];
}

/** *name*, cut to `MAX_NAME_CHARS`. */
function cutName(name: string): string {
  return name.length > MAX_NAME_CHARS ? `${name.slice(0, MAX_NAME_CHARS - 1)}…` : name;
}

/** The first line of an outline of *filePath*, as `readFacts` writes it. */
function outlineHeading(filePath: string): string {
  return `outline of ${filePath}:`;
}

/**
 * Whether *entries*, a session's context, hold a `read` result already
 * outlining *filePath*. The model has the outline while that result is
 * in its context, which follows `/tree` and compaction.
 */
export function outlinedIn(entries: readonly SessionEntry[], filePath: string): boolean {
  const heading = outlineHeading(filePath);
  return entries.some(
    (entry) =>
      entry.type === "message" &&
      entry.message.role === "toolResult" &&
      entry.message.toolName === "read" &&
      entry.message.content.some((part) => part.type === "text" && part.text.includes(heading)),
  );
}

/** The block to append, or null when there is nothing to say. */
export function block(lines: string[]): string | null {
  return lines.length === 0 ? null : `${HEADER}\n${lines.slice(0, MAX_LINES).join("\n")}`;
}
