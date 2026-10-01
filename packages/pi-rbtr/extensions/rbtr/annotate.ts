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

import type {
  FindRefsResponse,
  ListSymbolsResponse,
  ReadSymbolResponse,
  SearchResponse,
} from "./generated/protocol.js";

export const HEADER = "[rbtr index]";
const MAX_LINES = 10;
/** A read of a file shorter than this gets no outline: reading it whole is cheap. */
export const OUTLINE_FROM_LINES = 200;

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

function patternIn(args: string[]): string | null {
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--") return args[i + 1] ?? null;
    if (arg === "-e" || arg === "--regexp") return args[i + 1] ?? null;
    if (arg.startsWith("--regexp=")) return arg.slice("--regexp=".length);
    if (arg.startsWith("-")) {
      if (TAKES_VALUE.has(arg)) i++;
      continue;
    }
    return arg;
  }
  return null;
}

/** The pattern of the first `grep`, `rg`, `ag` or `git grep` in a command, if any. */
export function searchPattern(command: string): string | null {
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
  for (const [head, second, ...rest] of stages) {
    if (head === "git" && second === "grep") return patternIn(rest);
    if (head !== undefined && SEARCH_COMMANDS.has(head))
      return patternIn([second, ...rest].filter((w) => w !== undefined));
  }
  return null;
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
    const definitions = (await index.readSymbol(name)).chunks.slice(0, 3);
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
  const symbols = (await index.listSymbols(filePath)).chunks;
  if (symbols.length === 0) return [];
  const shown = symbols
    .slice(0, MAX_LINES - 2)
    .map((s) => `  ${s.name} (${s.kind}) lines ${s.line_start}-${s.line_end}`);
  const more = symbols.length > shown.length ? [`  +${symbols.length - shown.length} more: rbtr_list_symbols`] : [];
  return [`outline of ${filePath}:`, ...shown, ...more];
}

/** The block to append, or null when there is nothing to say. */
export function block(lines: string[]): string | null {
  return lines.length === 0 ? null : `${HEADER}\n${lines.slice(0, MAX_LINES).join("\n")}`;
}
