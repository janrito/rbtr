/**
 * The text the model reads for an rbtr reply, and the line shapes it
 * shares with the `[rbtr index]` facts. ARCHITECTURE, "Replies", gives
 * the reasons.
 */

import type {
  ChangedSymbolsResponse,
  ChunkKind,
  FindRefsResponse,
  ListSymbolsResponse,
  ReadSymbolResponse,
  ResolvedRef,
  Response,
  SearchResponse,
} from "./generated/protocol.js";
import { shortSha } from "./render.js";

/** Kinds an outline leaves out: they are not where a reader goes next. */
export const OUTLINE_EXCLUDES: [ChunkKind, ...ChunkKind[]] = ["import", "comment", "fragment"];
/** Lines of each search hit's body the model is sent; a longer hit says how many more it has. */
export const SEARCH_PREVIEW_LINES = 8;
/** An outline name longer than this is cut, so each line stays one short line. */
const MAX_NAME_CHARS = 80;

/** A symbol's name with its scope, as `rbtr_read_symbol` takes it: `Outer::Inner::name`. */
export function qualifiedName(symbol: { name: string; scope?: string }): string {
  return symbol.scope ? `${symbol.scope}::${symbol.name}` : symbol.name;
}

/** Where a symbol is, as one line's body: `path:line kind name`. */
export function located(path: string, line: number | string, kind: ChunkKind, name: string): string {
  return `${path}:${line} ${kind} ${name}`;
}

/** The first line of an outline of *filePath*. */
export function outlineHeading(filePath: string): string {
  return `outline of ${filePath}:`;
}

/** One symbol of an outline. */
export function outlineLine(symbol: {
  name: string;
  scope?: string;
  kind: ChunkKind;
  line_start: number;
  line_end: number;
}): string {
  const name = qualifiedName(symbol);
  const cut = name.length > MAX_NAME_CHARS ? `${name.slice(0, MAX_NAME_CHARS - 1)}…` : name;
  return `  ${cut} (${symbol.kind}) lines ${symbol.line_start}-${symbol.line_end}`;
}

/** The text the model reads for *response*. */
export function modelText(response: Response): string {
  switch (response.kind) {
    case "search":
      return searchText(response);
    case "read_symbol":
      return readSymbolText(response);
    case "list_symbols":
      return listSymbolsText(response);
    case "find_refs":
      return findRefsText(response);
    case "changed_symbols":
      return changedSymbolsText(response);
    default:
      return JSON.stringify(response);
  }
}

/** Which snapshot a read used, and how rbtr chose it. */
function readAt(resolved: ResolvedRef | null): string {
  return resolved ? `read at ${shortSha(resolved.sha)} (${resolved.source})` : "read at each repo's latest";
}

function searchText(response: SearchResponse): string {
  const lines = [`${readAt(response.resolved)}: ${response.results.length} result(s)`];
  for (const hit of response.results) {
    const repo = hit.repo_path ? `${hit.repo_path.replace(/\/+$/, "").split("/").pop()}/` : "";
    const paths = hit.file_paths.map((path) => `${repo}${path}`).join(", ");
    lines.push(`match: ${located(paths, hit.line_start, hit.kind, qualifiedName(hit))}`);
    lines.push(...hit.preview.text.split("\n").map((line) => `  ${line}`));
    if (hit.preview.clipped) {
      const shown = hit.preview.text.split("\n").length;
      lines.push(`  … ${hit.preview.total_lines - shown} more lines`);
    }
  }
  return lines.join("\n");
}

function readSymbolText(response: ReadSymbolResponse): string {
  if (response.chunks.length === 0) return `${readAt(response.resolved)}: no symbol by that name`;
  const lines = [readAt(response.resolved)];
  for (const chunk of response.chunks) {
    const range = `${chunk.line_start}-${chunk.line_end}`;
    lines.push(`defined: ${located(chunk.file_path, range, chunk.kind, qualifiedName(chunk))}`, chunk.content);
  }
  return lines.join("\n");
}

function listSymbolsText(response: ListSymbolsResponse): string {
  const heading = outlineHeading(response.file_path);
  if (response.chunks.length === 0) return `${readAt(response.resolved)}\n${heading} no symbols`;
  return [readAt(response.resolved), heading, ...response.chunks.map(outlineLine)].join("\n");
}

/** How many refs a find-refs reply lists; the rest are counted per kind. */
const MAX_REFS = 50;
const EDGE_ORDER = ["calls", "inherits", "imports", "documents", "configures"];

function findRefsText(response: FindRefsResponse): string {
  const byKind = new Map<string, FindRefsResponse["refs"]>();
  for (const ref of response.refs) byKind.set(ref.edge, [...(byKind.get(ref.edge) ?? []), ref]);
  const kinds = [...byKind.keys()].sort((a, b) => rank(a) - rank(b));
  // The space is shared in turn, so one kind's thousands of callers leave room for the others.
  const shown = new Map<string, number>(kinds.map((kind) => [kind, 0]));
  for (let listed = 0, round = 0; listed < MAX_REFS && listed < response.refs.length; round += 1) {
    for (const kind of kinds) {
      if (listed < MAX_REFS && round < (byKind.get(kind)?.length ?? 0)) {
        shown.set(kind, round + 1);
        listed += 1;
      }
    }
  }
  const lines = [`${readAt(response.resolved)}: ${response.refs.length} ref(s)`];
  for (const kind of kinds) {
    const refs = byKind.get(kind) ?? [];
    const count = shown.get(kind) ?? 0;
    lines.push(
      ...refs.slice(0, count).map((ref) => `${kind}: ${located(ref.file_path, ref.line_start, ref.kind, ref.name)}`),
    );
    if (refs.length > count) lines.push(`… ${refs.length - count} more ${kind}`);
  }
  return lines.join("\n");
}

/** A kind's place in a find-refs reply: callers first, unknown kinds last. */
function rank(kind: string): number {
  const position = EDGE_ORDER.indexOf(kind);
  return position === -1 ? EDGE_ORDER.length : position;
}

function changedSymbolsText(response: ChangedSymbolsResponse): string {
  return [
    `changes from ${shortSha(response.base_sha)} to ${shortSha(response.head_sha)}: ${response.changes.length} change(s)`,
    ...response.changes.map(
      ({ change, chunk }) =>
        `${change}: ${located(chunk.file_path, chunk.line_start, chunk.kind, qualifiedName(chunk))}`,
    ),
  ].join("\n");
}
