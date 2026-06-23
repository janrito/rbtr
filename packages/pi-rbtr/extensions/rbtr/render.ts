/**
 * Custom TUI renderers for rbtr tools.
 *
 * Each tool gets a compact renderCall (one-liner) and a
 * renderResult (collapsed/expanded views).
 *
 * Two sources of payload:
 *   - details.response — typed response from the daemon
 *                        (preferred path; no parsing).
 *   - content[].text   — one JSON response object, from the CLI
 *                        fallback path (the same shape, serialised).
 */

import type { AgentToolResult, Theme } from "@mariozechner/pi-coding-agent";
import { Text } from "@mariozechner/pi-tui";

import { decodeStringList } from "./args.js";

import type {
  ChangedSymbol,
  ChangedSymbolsResponse,
  FindRefsResponse,
  ListSymbolsResponse,
  ReadSymbolResponse,
  RefOut,
  Response,
  SearchHitOut,
  SearchResponse,
  StatusResponse,
  SymbolOut,
  WatchedRef,
} from "./generated/protocol.js";

type ToolResult = AgentToolResult<unknown>;

/**
 * Render the watch set as status lines (mirrors the Python
 * `_render_status_response` “watching:” section): ✓ indexed,
 * ⟳ pending (resolved but not yet indexed), ✗ unresolvable.
 * Returns no lines when nothing is watched.
 */
export function formatWatched(watched: WatchedRef[]): string[] {
  if (watched.length === 0) return [];
  const lines = ["Watching:"];
  for (const w of watched) {
    const repo = w.repo_path ? `${w.repo_path} ` : "";
    if (!w.sha) {
      lines.push(`  ✗ ${repo}${w.ref} — unresolvable`);
    } else if (w.indexed) {
      lines.push(`  ✓ ${repo}${w.ref} — ${w.sha.slice(0, 12)} indexed`);
    } else {
      lines.push(`  ⟳ ${repo}${w.ref} — ${w.sha.slice(0, 12)} pending`);
    }
  }
  return lines;
}

// ── Helpers ─────────────────────────────────────────────────────

function getContentText(result: ToolResult): string {
  for (const part of result.content) {
    if (part.type === "text" && "text" in part) return part.text;
  }
  return "";
}

function tryParseResponse(text: string): Response | undefined {
  try {
    return JSON.parse(text) as Response;
  } catch {
    return undefined;
  }
}

/**
 * Return the ``payloadKey`` array of the response for *responseKind*.
 *
 * Both transports carry one JSON response object: the daemon path
 * exposes it on ``result.details.response``; the CLI fallback prints it
 * to stdout (captured as the tool content text). Either way we narrow
 * the same generated ``Response`` union and read its list field.
 */
export function extractPayload<T>(result: ToolResult, responseKind: Response["kind"], payloadKey: string): T[] {
  const details = result.details as { fromDaemon?: boolean; response?: Response } | undefined;
  const response = details?.fromDaemon ? details.response : tryParseResponse(getContentText(result));
  if (response?.kind !== responseKind) return [];
  const value = (response as unknown as Record<string, unknown>)[payloadKey];
  return Array.isArray(value) ? (value as T[]) : [];
}

function shortenPath(filePath: string): string {
  // Remove common prefixes for readability
  return filePath.replace(/^packages\/rbtr\/src\//, "");
}

/**
 * Dim suffix describing a call's `file_paths` scoping, or `""` when the
 * call is unscoped.  Lets the call line distinguish a single-path lookup
 * from an everywhere lookup — otherwise the scoping argument is invisible.
 */
export function fileScopeSuffix(args: Record<string, unknown>, theme: Theme): string {
  const paths = decodeStringList(args.file_paths);
  if (paths.length === 0) return "";
  const label = paths.length === 1 ? shortenPath(paths[0]) : `${paths.length} files`;
  return theme.fg("dim", ` in ${label}`);
}

// ── Search ──────────────────────────────────────────────────────

function scoreStyle(theme: Theme, score: number): string {
  if (score >= 1.0) return theme.fg("success", score.toFixed(2));
  if (score >= 0.5) return theme.fg("warning", score.toFixed(2));
  return theme.fg("dim", score.toFixed(2));
}

function str(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return null;
}

function invalidArg(theme: Theme): string {
  return theme.fg("error", "[invalid arg]");
}

// Matched-query highlight reuses the `accent` role so the bit the
// user searched for stands out from the dim preview context.
const MATCH_STYLE = "accent" as const;

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Render *text* with *terms* highlighted; non-matches stay dim. */
function highlightTerms(theme: Theme, text: string, terms: string[]): string {
  if (terms.length === 0) return theme.fg("dim", text);
  // Longest-first so `agentdeps` wins over `agent` at the same index.
  const alts = [...terms].sort((a, b) => b.length - a.length).map(escapeRegExp);
  const re = new RegExp(`(${alts.join("|")})`, "gi");
  let out = "";
  let last = 0;
  for (const m of text.matchAll(re)) {
    const i = m.index ?? last;
    if (i > last) out += theme.fg("dim", text.slice(last, i));
    out += theme.fg(MATCH_STYLE, m[0]);
    last = i + m[0].length;
  }
  if (last < text.length) out += theme.fg("dim", text.slice(last));
  return out;
}

/**
 * Slice a preview window of up to *max* lines from *content*, scrolled
 * to *offset* only when it falls past the first window (mirrors the CLI
 * renderer).
 */
function previewWindow(
  content: string,
  offset: number | null | undefined,
  max: number,
): { window: string[]; start: number } {
  const all = content.split("\n");
  const start = offset != null && offset >= max ? offset - 1 : 0;
  return { window: all.slice(start, start + max), start };
}

export function renderSearchCall(args: Record<string, unknown>, theme: Theme): Text {
  const query = str(args.query);
  let text = theme.fg("toolTitle", theme.bold("rbtr_search "));
  text += query === null ? invalidArg(theme) : query ? theme.fg("accent", `"${query}"`) : theme.fg("toolOutput", "...");
  const extras: string[] = [];
  if (args.limit) extras.push(`limit: ${args.limit}`);
  const keywords = decodeStringList(args.keywords);
  if (keywords.length > 0) extras.push(`+${keywords.length} kw`);
  const variants = decodeStringList(args.variants);
  if (variants.length > 0) extras.push(`+${variants.length} var`);
  if (typeof args.scope === "string" && args.scope !== "workspace") extras.push(`scope: ${args.scope}`);
  if (extras.length > 0) text += theme.fg("dim", ` (${extras.join(", ")})`);
  return new Text(text, 0, 0);
}

export function renderSearchResult(
  result: ToolResult,
  options: { expanded: boolean; isPartial: boolean },
  theme: Theme,
): Text {
  if (options.isPartial) return new Text(theme.fg("warning", "Searching…"), 0, 0);

  const results = extractPayload<SearchHitOut>(result, "search", "results") satisfies SearchResponse["results"];
  if (results.length === 0) {
    return new Text(theme.fg("dim", "No results found."), 0, 0);
  }

  const lines: string[] = [];
  const show = options.expanded ? results : results.slice(0, 5);

  for (const r of show) {
    let path = shortenPath(r.file_path);
    if (r.repo_path) {
      const repoName = r.repo_path.replace(/\/+$/, "").split("/").pop() ?? r.repo_path;
      path = `${repoName}/${path}`;
    }
    let line = `${scoreStyle(theme, r.score)}  ${theme.fg("accent", path)}`;
    line += theme.fg("dim", `:${r.line_start}`);
    line += `  ${theme.fg("muted", r.kind)}  ${r.name}`;
    if (r.scope) line += theme.fg("dim", ` (${r.scope})`);
    lines.push(line);

    const terms = r.matched_terms ?? [];
    if (options.expanded) {
      const { window, start } = previewWindow(r.content, r.match_line_offset, 4);
      if (start > 0) {
        // Show the chunk's signature line for orientation, then the gap.
        const signature = r.content.split("\n")[0] ?? "";
        lines.push(`  ${highlightTerms(theme, signature, terms)}`);
        lines.push(theme.fg("dim", "  …"));
      }
      for (const pl of window) {
        lines.push(`  ${highlightTerms(theme, pl, terms)}`);
      }
      lines.push("");
    } else if (r.match_line_offset != null) {
      // Surface the matched line so the collapsed view shows “that
      // bit” without expanding.
      const anchorLine = r.content.split("\n")[r.match_line_offset] ?? "";
      lines.push(`  ${highlightTerms(theme, anchorLine, terms)}`);
    }
  }

  if (!options.expanded && results.length > 5) {
    lines.push(theme.fg("muted", `… ${results.length - 5} more results`));
  }

  lines.push(theme.fg("dim", `${results.length} result(s)`));
  return new Text(lines.join("\n"), 0, 0);
}

// ── Read symbol ─────────────────────────────────────────────────

export function renderReadSymbolCall(args: Record<string, unknown>, theme: Theme): Text {
  const symbol = str(args.symbol);
  const label =
    symbol === null ? invalidArg(theme) : symbol ? theme.fg("accent", symbol) : theme.fg("toolOutput", "...");
  return new Text(theme.fg("toolTitle", theme.bold("rbtr_read_symbol ")) + label + fileScopeSuffix(args, theme), 0, 0);
}

export function renderReadSymbolResult(
  result: ToolResult,
  options: { expanded: boolean; isPartial: boolean },
  theme: Theme,
): Text {
  if (options.isPartial) return new Text(theme.fg("warning", "Reading…"), 0, 0);

  const chunks = extractPayload<SymbolOut>(result, "read_symbol", "chunks") satisfies ReadSymbolResponse["chunks"];
  const symbol = (result.details as { symbol?: string } | undefined)?.symbol;
  if (chunks.length === 0) {
    return new Text(theme.fg("error", `Symbol not found${symbol ? `: ${symbol}` : ""}`), 0, 0);
  }

  const lines: string[] = [];
  for (const c of chunks) {
    const path = shortenPath(c.file_path);
    lines.push(
      `${theme.fg("accent", path)}${theme.fg("dim", `:${c.line_start}-${c.line_end}`)}  ${theme.fg("muted", c.kind)}  ${c.name}`,
    );

    if (options.expanded) {
      for (const cl of c.content.split("\n")) {
        lines.push(`  ${theme.fg("dim", cl)}`);
      }
      lines.push("");
    }
  }

  if (!options.expanded && chunks.length > 0) {
    const first = chunks[0];
    const lineCount = first.content.split("\n").length;
    lines.push(theme.fg("dim", `${lineCount} lines`));
  }

  return new Text(lines.join("\n"), 0, 0);
}

// ── List symbols ────────────────────────────────────────────────

export function renderListSymbolsCall(args: Record<string, unknown>, theme: Theme): Text {
  const file = str(args.file);
  const label = file === null ? invalidArg(theme) : file ? theme.fg("accent", file) : theme.fg("toolOutput", "...");
  return new Text(theme.fg("toolTitle", theme.bold("rbtr_list_symbols ")) + label, 0, 0);
}

export function renderListSymbolsResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
  if (options.isPartial) return new Text(theme.fg("warning", "Listing…"), 0, 0);

  const chunks = extractPayload<SymbolOut>(result, "list_symbols", "chunks") satisfies ListSymbolsResponse["chunks"];
  if (chunks.length === 0) {
    return new Text(theme.fg("dim", "No symbols found."), 0, 0);
  }

  const lines: string[] = [];
  for (const c of chunks) {
    const range = `${String(c.line_start).padStart(4)}-${String(c.line_end).padEnd(4)}`;
    lines.push(`${theme.fg("dim", range)}  ${theme.fg("muted", c.kind.padEnd(15))}${c.name}`);
  }
  lines.push(theme.fg("dim", `${chunks.length} symbol(s)`));
  return new Text(lines.join("\n"), 0, 0);
}

// ── Find refs ───────────────────────────────────────────────────

export function renderFindRefsCall(args: Record<string, unknown>, theme: Theme): Text {
  const symbol = str(args.symbol);
  const label =
    symbol === null ? invalidArg(theme) : symbol ? theme.fg("accent", symbol) : theme.fg("toolOutput", "...");
  return new Text(theme.fg("toolTitle", theme.bold("rbtr_find_refs ")) + label + fileScopeSuffix(args, theme), 0, 0);
}

export function renderFindRefsResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
  if (options.isPartial) return new Text(theme.fg("warning", "Finding…"), 0, 0);

  const refs = extractPayload<RefOut>(result, "find_refs", "refs") satisfies FindRefsResponse["refs"];
  if (refs.length === 0) {
    return new Text(theme.fg("dim", "No references found."), 0, 0);
  }

  const lines: string[] = [];
  for (const r of refs) {
    const path = shortenPath(r.file_path);
    lines.push(
      `${theme.fg("accent", path)}${theme.fg("dim", `:${r.line_start}`)}  ${theme.fg("muted", r.kind)}  ${r.name}  ${theme.fg("dim", `(${r.edge})`)}`,
    );
  }
  lines.push(theme.fg("dim", `${refs.length} reference(s)`));
  return new Text(lines.join("\n"), 0, 0);
}

// ── Changed symbols ─────────────────────────────────────────────

export function renderChangedSymbolsCall(args: Record<string, unknown>, theme: Theme): Text {
  const base = str(args.base);
  const head = str(args.head);
  const label = base && head ? theme.fg("accent", `${base}..${head}`) : invalidArg(theme);
  return new Text(
    theme.fg("toolTitle", theme.bold("rbtr_changed_symbols ")) + label + fileScopeSuffix(args, theme),
    0,
    0,
  );
}

export function renderChangedSymbolsResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
  if (options.isPartial) return new Text(theme.fg("warning", "Comparing…"), 0, 0);

  const changes = extractPayload<ChangedSymbol>(
    result,
    "changed_symbols",
    "changes",
  ) satisfies ChangedSymbolsResponse["changes"];
  if (changes.length === 0) {
    return new Text(theme.fg("dim", "No changed symbols."), 0, 0);
  }

  // Shared vocabulary with the CLI: sigils + ordering + summary.
  const sigil = { added: "+", modified: "~", removed: "\u2212" } as const;
  const colour = { added: "toolDiffAdded", modified: "warning", removed: "toolDiffRemoved" } as const;
  const order = { added: 0, modified: 1, removed: 2 } as const;

  const counts = { added: 0, modified: 0, removed: 0 };
  for (const ch of changes) counts[ch.change]++;

  const sorted = [...changes].sort((a, b) => order[a.change] - order[b.change]);
  const cap = 8;
  const shown = sorted.slice(0, cap);

  const lines: string[] = [];
  for (const ch of shown) {
    const c = ch.chunk;
    const path = shortenPath(c.file_path);
    lines.push(
      `${theme.fg(colour[ch.change], sigil[ch.change])} ${theme.fg("muted", c.kind.padEnd(10))}${c.name}  ${theme.fg("dim", path)}`,
    );
  }
  if (sorted.length > cap) {
    lines.push(theme.fg("muted", `\u2026 ${sorted.length - cap} more`));
  }
  lines.push(
    `${theme.fg("toolDiffAdded", `+${counts.added}`)}  ${theme.fg("warning", `~${counts.modified}`)}  ${theme.fg("toolDiffRemoved", `\u2212${counts.removed}`)}`,
  );
  return new Text(lines.join("\n"), 0, 0);
}

// ── Index ───────────────────────────────────────────────────────

export function renderIndexCall(args: Record<string, unknown>, theme: Theme): Text {
  const refs = decodeStringList(args.refs);
  const label = refs.length > 0 ? refs.join(", ") : "HEAD";
  return new Text(theme.fg("toolTitle", theme.bold("rbtr_index ")) + theme.fg("accent", label), 0, 0);
}

export function renderIndexResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
  if (options.isPartial) return new Text(theme.fg("muted", "Indexing…"), 0, 0);

  const details = result.details as { status?: string } | undefined;
  switch (details?.status) {
    case "started":
      return new Text(theme.fg("success", "✓ Indexing queued"), 0, 0);
    case "up_to_date":
      return new Text(theme.fg("success", "✓ Index up to date"), 0, 0);
    case "in_progress":
      return new Text(theme.fg("muted", "⟳ Build already in progress"), 0, 0);
  }

  const text = getContentText(result);
  return new Text(theme.fg("dim", text), 0, 0);
}

// ── Status ──────────────────────────────────────────────────────

export function renderStatusCall(_args: Record<string, unknown>, theme: Theme): Text {
  return new Text(theme.fg("toolTitle", theme.bold("rbtr_status")), 0, 0);
}

// Output is derived solely from the response model — no external state.
export function renderStatusResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
  if (options.isPartial) return new Text(theme.fg("muted", "Checking…"), 0, 0);

  // Daemon path packs a StatusResponse on details.response.
  const details = result.details as { fromDaemon?: boolean; response?: StatusResponse } | undefined;
  const response = details?.fromDaemon ? details.response : undefined;

  const lines: string[] = [];
  const indexed = response?.indexed_refs ?? [];
  const crossRepo = new Set(indexed.map((ref) => ref.repo_path)).size > 1;
  if (indexed.length === 0) {
    lines.push(theme.fg("error", "✗ No index found"));
  } else if (crossRepo) {
    lines.push(theme.fg("success", "✓ indexed repos"));
    const byRepo = new Map<string, typeof indexed>();
    for (const ref of indexed) {
      const key = ref.repo_path ?? "?";
      const group = byRepo.get(key) ?? [];
      group.push(ref);
      byRepo.set(key, group);
    }
    for (const [repoPath, refs] of byRepo) {
      lines.push(theme.fg("accent", repoPath));
      for (const ref of refs) {
        lines.push(`  ${theme.fg("muted", fmtRefRender(ref))}`);
      }
    }
  } else {
    const total = indexed[0].total;
    lines.push(theme.fg("success", `✓ ${humanCountRender(total)} symbols`));
    for (const ref of indexed) {
      lines.push(theme.fg("muted", fmtRefRender(ref)));
    }
  }

  const job = response?.active_build;
  if (job) {
    const pct = job.total > 0 ? ` (${Math.round((100 * job.current) / job.total)}%)` : "";
    const elapsed = formatElapsedRender(job.elapsed_seconds);
    lines.push(
      theme.fg(
        "muted",
        `⟳ Building: ${job.ref.slice(0, 12)} — ${job.phase} ${job.current}/${job.total}${pct} — ${elapsed}`,
      ),
    );
  }
  const ej = response?.active_embed;
  if (ej) {
    const pct = ej.total > 0 ? ` (${Math.round((100 * ej.current) / ej.total)}%)` : "";
    const elapsed = formatElapsedRender(ej.elapsed_seconds);
    lines.push(
      theme.fg(
        "muted",
        `\u21BB Embedding: ${ej.ref.slice(0, 12)} \u2014 ${ej.current}/${ej.total}${pct} \u2014 ${elapsed}`,
      ),
    );
  }

  return new Text(lines.join("\n"), 0, 0);
}

function formatElapsedRender(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m${String(s).padStart(2, "0")}s`;
}

function humanCountRender(n: number): string {
  if (n < 1000) return String(n);
  return `${(n / 1000).toFixed(1)}k`;
}

function fmtRefRender(ref: { sha: string; names?: string[]; total: number; embedded: number }): string {
  const label =
    (ref.names ?? []).length > 0 ? `${ref.sha.slice(0, 12)} (${(ref.names ?? []).join(", ")})` : ref.sha.slice(0, 12);
  const embedPart =
    ref.embedded >= ref.total
      ? `${humanCountRender(ref.embedded)} embedded \u2713`
      : ref.embedded > 0
        ? `${humanCountRender(ref.embedded)} embedded (${Math.round((100 * ref.embedded) / ref.total)}%)`
        : "not embedded";
  return `${label}  ${humanCountRender(ref.total)} indexed  ${embedPart}`;
}
