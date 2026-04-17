/**
 * Custom TUI renderers for rbtr tools.
 *
 * Each tool gets a compact renderCall (one-liner) and a
 * renderResult (collapsed/expanded views).
 *
 * Two sources of payload:
 *   - details.response        — typed response from the daemon
 *                               (preferred path; no parsing).
 *   - content[].text (NDJSON) — from the CLI fallback path.
 */

import type { AgentToolResult, Theme } from "@mariozechner/pi-coding-agent";
import { Text } from "@mariozechner/pi-tui";

import type {
	ChangedSymbolsResponse,
	Chunk,
	Edge,
	FindRefsResponse,
	ListSymbolsResponse,
	ReadSymbolResponse,
	Response,
	ScoredResult,
	SearchResponse,
	StatusResponse,
} from "./generated/protocol.js";

type ToolResult = AgentToolResult<unknown>;

// ── Helpers ─────────────────────────────────────────────────────

function getContentText(result: ToolResult): string {
	for (const part of result.content) {
		if (part.type === "text" && "text" in part) return part.text;
	}
	return "";
}

function tryParseNdjson<T>(text: string): T[] {
	try {
		return text
			.split("\n")
			.filter((line) => line.trim())
			.map((line) => JSON.parse(line) as T);
	} catch {
		return [];
	}
}

/**
 * Read the typed daemon response off ``result.details`` if it
 * came from the daemon path; otherwise parse NDJSON stdout.
 */
function extractPayload<T>(result: ToolResult, responseKind: Response["kind"], payloadKey: string): T[] {
	const details = result.details as { fromDaemon?: boolean; response?: Response } | undefined;
	if (details?.fromDaemon && details.response?.kind === responseKind) {
		const value = (details.response as unknown as Record<string, unknown>)[payloadKey];
		return Array.isArray(value) ? (value as T[]) : [];
	}
	return tryParseNdjson<T>(getContentText(result));
}

function shortenPath(filePath: string): string {
	// Remove common prefixes for readability
	return filePath.replace(/^packages\/rbtr\/src\//, "").replace(/^packages\/rbtr-legacy\/src\//, "legacy/");
}

// ── Search ──────────────────────────────────────────────────────

function scoreStyle(theme: Theme, score: number): string {
	if (score >= 1.0) return theme.fg("success", score.toFixed(2));
	if (score >= 0.5) return theme.fg("warning", score.toFixed(2));
	return theme.fg("dim", score.toFixed(2));
}

export function renderSearchCall(args: Record<string, unknown>, theme: Theme): Text {
	let text = theme.fg("toolTitle", theme.bold("rbtr_search "));
	text += theme.fg("accent", `"${args.query}"`);
	if (args.limit) text += theme.fg("dim", ` (limit: ${args.limit})`);
	return new Text(text, 0, 0);
}

export function renderSearchResult(
	result: ToolResult,
	options: { expanded: boolean; isPartial: boolean },
	theme: Theme,
): Text {
	if (options.isPartial) return new Text(theme.fg("warning", "Searching…"), 0, 0);

	const results = extractPayload<ScoredResult>(result, "search", "results") satisfies SearchResponse["results"];
	if (results.length === 0) {
		return new Text(theme.fg("dim", "No results found."), 0, 0);
	}

	const lines: string[] = [];
	const show = options.expanded ? results : results.slice(0, 5);

	for (const r of show) {
		const c = r.chunk;
		const path = shortenPath(c.file_path);
		let line = `${scoreStyle(theme, r.score)}  ${theme.fg("accent", path)}`;
		line += theme.fg("dim", `:${c.line_start}`);
		line += `  ${theme.fg("muted", c.kind)}  ${c.name}`;
		if (c.scope) line += theme.fg("dim", ` (${c.scope})`);
		lines.push(line);

		if (options.expanded) {
			const preview = c.content.split("\n").slice(0, 4).join("\n");
			for (const pl of preview.split("\n")) {
				lines.push(`  ${theme.fg("dim", pl)}`);
			}
			lines.push("");
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
	return new Text(
		theme.fg("toolTitle", theme.bold("rbtr_read_symbol ")) + theme.fg("accent", String(args.symbol)),
		0,
		0,
	);
}

export function renderReadSymbolResult(
	result: ToolResult,
	options: { expanded: boolean; isPartial: boolean },
	theme: Theme,
): Text {
	if (options.isPartial) return new Text(theme.fg("warning", "Reading…"), 0, 0);

	const chunks = extractPayload<Chunk>(result, "read_symbol", "chunks") satisfies ReadSymbolResponse["chunks"];
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
	return new Text(
		theme.fg("toolTitle", theme.bold("rbtr_list_symbols ")) + theme.fg("accent", String(args.file)),
		0,
		0,
	);
}

export function renderListSymbolsResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
	if (options.isPartial) return new Text(theme.fg("warning", "Listing…"), 0, 0);

	const chunks = extractPayload<Chunk>(result, "list_symbols", "chunks") satisfies ListSymbolsResponse["chunks"];
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
	return new Text(theme.fg("toolTitle", theme.bold("rbtr_find_refs ")) + theme.fg("accent", String(args.symbol)), 0, 0);
}

export function renderFindRefsResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
	if (options.isPartial) return new Text(theme.fg("warning", "Finding…"), 0, 0);

	const edges = extractPayload<Edge>(result, "find_refs", "edges") satisfies FindRefsResponse["edges"];
	if (edges.length === 0) {
		return new Text(theme.fg("dim", "No references found."), 0, 0);
	}

	const lines: string[] = [];
	for (const e of edges) {
		lines.push(`${e.source_id} ${theme.fg("dim", "→")} ${e.target_id}  ${theme.fg("muted", `(${e.kind})`)}`);
	}
	lines.push(theme.fg("dim", `${edges.length} reference(s)`));
	return new Text(lines.join("\n"), 0, 0);
}

// ── Changed symbols ─────────────────────────────────────────────

export function renderChangedSymbolsCall(args: Record<string, unknown>, theme: Theme): Text {
	return new Text(
		theme.fg("toolTitle", theme.bold("rbtr_changed_symbols ")) + theme.fg("accent", `${args.base}..${args.head}`),
		0,
		0,
	);
}

export function renderChangedSymbolsResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
	if (options.isPartial) return new Text(theme.fg("warning", "Comparing…"), 0, 0);

	const chunks = extractPayload<Chunk>(result, "changed_symbols", "chunks") satisfies ChangedSymbolsResponse["chunks"];
	if (chunks.length === 0) {
		return new Text(theme.fg("dim", "No changed symbols."), 0, 0);
	}

	const lines: string[] = [];
	for (const c of chunks) {
		const path = shortenPath(c.file_path);
		lines.push(`${theme.fg("muted", c.kind.padEnd(15))}${c.name}  ${theme.fg("dim", path)}`);
	}
	lines.push(theme.fg("dim", `${chunks.length} changed symbol(s)`));
	return new Text(lines.join("\n"), 0, 0);
}

// ── Index ───────────────────────────────────────────────────────

export function renderIndexCall(args: Record<string, unknown>, theme: Theme): Text {
	const refs = (args.refs as string[]) || ["HEAD"];
	return new Text(theme.fg("toolTitle", theme.bold("rbtr_index ")) + theme.fg("accent", refs.join(", ")), 0, 0);
}

export function renderIndexResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
	if (options.isPartial) return new Text(theme.fg("warning", "Indexing…"), 0, 0);

	const details = result.details as { status?: string } | undefined;
	if (details?.status === "started") {
		return new Text(theme.fg("success", "✓ Indexing started"), 0, 0);
	}

	const text = getContentText(result);
	return new Text(theme.fg("dim", text), 0, 0);
}

// ── Status ──────────────────────────────────────────────────────

export function renderStatusCall(_args: Record<string, unknown>, theme: Theme): Text {
	return new Text(theme.fg("toolTitle", theme.bold("rbtr_status")), 0, 0);
}

export function renderStatusResult(result: ToolResult, options: { isPartial: boolean }, theme: Theme): Text {
	if (options.isPartial) return new Text(theme.fg("warning", "Checking…"), 0, 0);

	// Daemon path packs a StatusResponse on details.response.
	const details = result.details as
		| { fromDaemon?: boolean; response?: StatusResponse; exists?: boolean; total_chunks?: number }
		| undefined;
	const response = details?.fromDaemon ? details.response : undefined;
	const exists = response?.exists ?? details?.exists ?? false;
	const total = response?.total_chunks ?? details?.total_chunks ?? 0;

	if (exists) {
		return new Text(theme.fg("success", `✓ ${total} symbols`), 0, 0);
	}
	return new Text(theme.fg("error", "✗ No index found"), 0, 0);
}
