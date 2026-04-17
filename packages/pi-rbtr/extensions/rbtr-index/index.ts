/**
 * rbtr-index — pi extension for the rbtr structural code index.
 *
 * Gives the LLM access to rbtr's code index via registered tools.
 * Tries the ZMQ daemon first; falls back to shelling the CLI on
 * transport errors.  Protocol types come from the generated
 * `./generated/protocol.ts` — Python is the source of truth.
 *
 * Placement: .pi/extensions/rbtr-index/index.ts
 */

import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	getSettingsListTheme,
	truncateHead,
} from "@mariozechner/pi-coding-agent";
import { Container, type SettingItem, SettingsList } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

import { RbtrDaemonError } from "./daemon-client.js";
import { DaemonSession, DaemonUnavailableError } from "./daemon-session.js";
import { type ResolvedCommand, resolveCommand, runRbtr, runRbtrJson } from "./exec.js";
import type { BuildIndexResponse, Response, StatusResponse } from "./generated/protocol.js";
import {
	renderChangedSymbolsCall,
	renderChangedSymbolsResult,
	renderFindRefsCall,
	renderFindRefsResult,
	renderIndexCall,
	renderIndexResult,
	renderListSymbolsCall,
	renderListSymbolsResult,
	renderReadSymbolCall,
	renderReadSymbolResult,
	renderSearchCall,
	renderSearchResult,
	renderStatusCall,
	renderStatusResult,
} from "./render.js";
import { loadSettings, type RbtrIndexSettings, saveProjectSettings } from "./settings.js";

// ── Tool result shape ─────────────────────────────────────────

interface ToolReturn {
	content: Array<{ type: "text"; text: string }>;
	details: Record<string, unknown>;
}

/** Pack a typed daemon response for the LLM + renderer. */
function toolResultFromDaemon(response: Response): ToolReturn {
	return {
		content: [{ type: "text", text: JSON.stringify(response) }],
		details: { fromDaemon: true, response },
	};
}

/** Pack raw CLI stdout for the LLM + renderer. */
function toolResultFromCli(stdout: string, extra: Record<string, unknown>): ToolReturn {
	const truncation = truncateHead(stdout, {
		maxLines: DEFAULT_MAX_LINES,
		maxBytes: DEFAULT_MAX_BYTES,
	});
	let content = truncation.content;
	if (truncation.truncated) {
		content +=
			`\n\n[Output truncated: showing ${truncation.outputLines} of ` +
			`${truncation.totalLines} lines. Use --limit or rbtr_read_symbol for details.]`;
	}
	return {
		content: [{ type: "text", text: content }],
		details: { fromCli: true, truncated: truncation.truncated, ...extra },
	};
}

// ── Extension ─────────────────────────────────────────────────

export default function rbtrIndexExtension(pi: ExtensionAPI) {
	const session = new DaemonSession();
	let resolved: ResolvedCommand | null = null;
	let settings: RbtrIndexSettings = { command: "rbtr", autoIndex: true };
	let cliAvailable = false;

	/**
	 * Try the daemon path, fall back to the CLI callback on
	 * transport failure.  Propagates ``RbtrDaemonError`` from the
	 * daemon untouched — that is an actionable reply, not a
	 * transport problem.
	 */
	async function withFallback<T>(fromDaemon: () => Promise<T>, fromCli: () => Promise<T>): Promise<T> {
		if (session.available) {
			try {
				return await fromDaemon();
			} catch (err) {
				if (err instanceof RbtrDaemonError) throw err;
				if (err instanceof DaemonUnavailableError) {
					// Transport failure — fall through to CLI.
				} else {
					throw err;
				}
			}
		}
		if (!resolved || !cliAvailable) {
			throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
		}
		return fromCli();
	}

	function mapDaemonError(err: unknown): ToolReturn {
		if (err instanceof RbtrDaemonError) {
			return {
				content: [{ type: "text", text: `${err.code}: ${err.message}` }],
				details: { errorCode: err.code, message: err.message },
			};
		}
		throw err;
	}

	async function queryIndexStatus(repo: string): Promise<StatusResponse | null> {
		if (session.available) {
			try {
				return await session.send({ kind: "status", repo });
			} catch (err) {
				if (err instanceof RbtrDaemonError) return null;
				// transport — fall through to CLI
			}
		}
		if (!resolved) return null;
		try {
			const results = await runRbtrJson<StatusResponse>(pi, resolved, ["status"], { timeout: 5000 });
			return results[0] ?? null;
		} catch {
			return null;
		}
	}

	// ── Session lifecycle ───────────────────────────────────────

	pi.on("session_start", async (_event, ctx) => {
		settings = loadSettings(ctx.cwd);
		resolved = resolveCommand(settings.command);

		// Look for the daemon first — one CLI shell-out here avoids
		// a per-tool-call query.  Failure leaves the session marked
		// unavailable and the tools fall back to CLI exec.
		await session.refresh();

		const status = await queryIndexStatus(ctx.cwd);
		if (status === null) {
			cliAvailable = false;
			ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("error", "rbtr: not found"));
			ctx.ui.notify(
				"rbtr CLI not found. Install with: uv tool install rbtr\n" +
					'Or set command in .pi/rbtr-index.json to "uvx" or "uvx --from <path>"',
				"warning",
			);
			return;
		}

		cliAvailable = true;

		if (status.exists) {
			ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("success", `rbtr: ${status.total_chunks} symbols`));
		} else if (settings.autoIndex) {
			await triggerIndex(ctx);
		} else {
			ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("warning", "rbtr: no index \u2014 /rbtr-index"));
		}
	});

	pi.on("before_agent_start", async (event) => {
		if (!cliAvailable) return;
		return {
			systemPrompt:
				event.systemPrompt +
				"\n\nThe rbtr code index is available for this repository. " +
				"Use rbtr_search for concept queries (more precise than grep for semantic searches), " +
				"rbtr_read_symbol to read a symbol's full source by name, " +
				"rbtr_list_symbols for a file's structural table of contents.",
		};
	});

	/**
	 * Submit a build to the daemon (or via CLI fallback).
	 *
	 * Fire-and-forget: the daemon returns OkResponse immediately
	 * and runs the build on its internal queue.  Progress updates
	 * arrive via PUB notifications (wired in Phase 8.4).
	 */
	async function triggerIndex(ctx: ExtensionContext, ...refs: string[]): Promise<void> {
		const targetRefs = refs.length > 0 ? refs : ["HEAD"];
		ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("warning", "rbtr: indexing\u2026"));
		try {
			await withFallback(
				async () => {
					await session.send({ kind: "index", repo: ctx.cwd, refs: targetRefs });
				},
				async () => {
					if (!resolved) throw new Error("rbtr CLI not available");
					const args = ["index"];
					for (const r of targetRefs) args.push(r);
					await runRbtrJson<BuildIndexResponse>(pi, resolved, args, { timeout: 600_000 });
				},
			);
		} catch (err) {
			ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("error", "rbtr: indexing failed"));
			ctx.ui.notify(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`, "error");
		}
	}

	// ── Commands ────────────────────────────────────────────────

	pi.registerCommand("rbtr-status", {
		description: "Show rbtr index status",
		handler: async (_args, ctx) => {
			if (!cliAvailable) {
				ctx.ui.notify("rbtr CLI not available", "error");
				return;
			}
			const status = await queryIndexStatus(ctx.cwd);
			if (!status) {
				ctx.ui.notify("Failed to get index status", "error");
				return;
			}
			if (status.exists) {
				ctx.ui.notify(`Index: ${status.total_chunks} symbols\nPath: ${status.db_path}`, "info");
			} else {
				ctx.ui.notify("No index found. Use /rbtr-index to create one.", "warning");
			}
		},
	});

	pi.registerCommand("rbtr-index", {
		description: "Index the repository (or rebuild the index)",
		handler: async (_args, ctx) => {
			if (!cliAvailable) {
				ctx.ui.notify("rbtr CLI not available", "error");
				return;
			}
			await triggerIndex(ctx);
			ctx.ui.notify("Indexing started. Progress in the footer.", "info");
		},
	});

	pi.registerCommand("rbtr-settings", {
		description: "View and toggle rbtr index settings",
		handler: async (_args, ctx) => {
			if (!ctx.hasUI) {
				ctx.ui.notify("/rbtr-settings requires interactive mode", "error");
				return;
			}

			await ctx.ui.custom((tui, theme, _kb, done) => {
				const items: SettingItem[] = [
					{
						id: "autoIndex",
						label: "Auto-index on session start",
						currentValue: settings.autoIndex ? "on" : "off",
						values: ["on", "off"],
					},
				];

				const container = new Container();

				container.addChild({
					render(_width: number) {
						const desc = resolved ? resolved.description : "not resolved";
						const daemonMode = session.available ? "daemon" : "cli fallback";
						return [
							theme.fg("accent", theme.bold("rbtr Index Settings")),
							"",
							`${theme.fg("muted", "Command:")} ${settings.command}`,
							`${theme.fg("muted", "Resolved:")} ${theme.fg("dim", desc)}`,
							`${theme.fg("muted", "CLI available:")} ${cliAvailable ? theme.fg("success", "yes") : theme.fg("error", "no")}`,
							`${theme.fg("muted", "Transport:")} ${theme.fg(session.available ? "success" : "warning", daemonMode)}`,
							"",
						];
					},
					invalidate() {},
				});

				const settingsList = new SettingsList(
					items,
					Math.min(items.length + 2, 15),
					getSettingsListTheme(),
					(id, newValue) => {
						if (id === "autoIndex") {
							settings.autoIndex = newValue === "on";
							saveProjectSettings(ctx.cwd, { autoIndex: settings.autoIndex });
						}
					},
					() => done(undefined),
				);

				container.addChild(settingsList);

				return {
					render(width: number) {
						return container.render(width);
					},
					invalidate() {
						container.invalidate();
					},
					handleInput(data: string) {
						settingsList.handleInput?.(data);
						tui.requestRender();
					},
				};
			});
		},
	});

	// ── Tools ──────────────────────────────────────────────────

	pi.registerTool({
		name: "rbtr_index",
		label: "rbtr index",
		description: "Index the repository so the code index tools can be used.",
		promptSnippet: "Index the repository so the code index tools can be used",
		promptGuidelines: [
			"Use rbtr_index when the user asks to index the codebase or when rbtr_search returns no results.",
			"Indexing is incremental — unchanged files are skipped.",
			"The index runs in background. Use rbtr_status to check when it completes.",
		],
		parameters: Type.Object({
			refs: Type.Optional(
				Type.Array(Type.String(), { description: "Git refs to index (default: ['HEAD']). Two refs = base + head." }),
			),
		}),
		renderCall: (args, theme) => renderIndexCall(args, theme),
		renderResult: (result, options, theme) => renderIndexResult(result, options, theme),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			if (!cliAvailable) {
				throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
			}
			const refs = params.refs ?? ["HEAD"];
			await triggerIndex(ctx, ...refs);
			return {
				content: [
					{
						type: "text",
						text: `Indexing started for refs ${refs.join(", ")}. Progress is shown in the footer. Use rbtr_status to check when complete.`,
					},
				],
				details: { status: "started", refs },
			};
		},
	});

	pi.registerTool({
		name: "rbtr_status",
		label: "rbtr status",
		description: "Check whether the code index exists and how many symbols it contains.",
		promptSnippet: "Check whether the code index exists and how many symbols it contains",
		parameters: Type.Object({}),
		renderCall: (args, theme) => renderStatusCall(args, theme),
		renderResult: (result, options, theme) => renderStatusResult(result, options, theme),

		async execute(_toolCallId, _params, _signal, _onUpdate, ctx) {
			if (!cliAvailable) {
				throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
			}
			const status = await queryIndexStatus(ctx.cwd);
			if (!status) {
				throw new Error("Failed to check index status");
			}
			if (status.exists) {
				return {
					content: [{ type: "text", text: `Index: ${status.total_chunks} symbols\nPath: ${status.db_path}` }],
					details: { fromDaemon: true, response: status },
				};
			}
			return {
				content: [{ type: "text", text: "No index found. Use rbtr_index to create one." }],
				details: { fromDaemon: true, response: status },
			};
		},
	});

	pi.registerTool({
		name: "rbtr_search",
		label: "rbtr search",
		description: "Search the structural code index for symbols, functions, classes, and code patterns.",
		promptSnippet: "Search the structural code index for symbols, functions, classes, and code patterns",
		promptGuidelines: [
			"Use rbtr_search for conceptual queries like 'retry logic' or 'error handling'. For exact string matches, use grep instead.",
			"Search results include score breakdown. Higher scores indicate better matches.",
			"Use rbtr_read_symbol to read the full source of a result.",
		],
		parameters: Type.Object({
			query: Type.String({ description: "Search query" }),
			limit: Type.Optional(Type.Number({ description: "Maximum results to return (default: 10)" })),
		}),
		renderCall: (args, theme) => renderSearchCall(args, theme),
		renderResult: (result, options, theme) => renderSearchResult(result, options, theme),

		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			try {
				return await withFallback<ToolReturn>(
					async () => {
						const resp = await session.send({
							kind: "search",
							repo: ctx.cwd,
							query: params.query,
							...(params.limit !== undefined ? { limit: params.limit } : {}),
						});
						if (resp.results.length === 0) {
							return {
								content: [{ type: "text", text: "No results found." }],
								details: { fromDaemon: true, response: resp },
							};
						}
						return toolResultFromDaemon(resp);
					},
					async () => {
						if (!resolved) throw new Error("rbtr CLI not available");
						const args = ["search", params.query];
						if (params.limit !== undefined) args.push("--limit", String(params.limit));
						const result = await runRbtr(pi, resolved, args, { signal, timeout: 30_000 });
						const text = result.stdout.trim();
						if (!text) {
							return {
								content: [{ type: "text", text: "No results found." }],
								details: { fromCli: true, results: [] },
							};
						}
						return toolResultFromCli(text, { query: params.query, limit: params.limit });
					},
				);
			} catch (err) {
				return mapDaemonError(err);
			}
		},
	});

	pi.registerTool({
		name: "rbtr_read_symbol",
		label: "rbtr read-symbol",
		description: "Read a symbol's full source code by name from the code index.",
		promptSnippet: "Read a symbol's full source code by name from the code index",
		promptGuidelines: [
			"Use rbtr_read_symbol after rbtr_search to read the full source of a specific symbol.",
			"More precise than grep or read for navigating to a known symbol by name.",
		],
		parameters: Type.Object({
			symbol: Type.String({ description: "Symbol name (e.g. HttpClient.retry, fuse_scores)" }),
		}),
		renderCall: (args, theme) => renderReadSymbolCall(args, theme),
		renderResult: (result, options, theme) => renderReadSymbolResult(result, options, theme),

		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			try {
				return await withFallback<ToolReturn>(
					async () => {
						const resp = await session.send({
							kind: "read_symbol",
							repo: ctx.cwd,
							name: params.symbol,
						});
						if (resp.chunks.length === 0) {
							return {
								content: [{ type: "text", text: `Symbol not found: ${params.symbol}` }],
								details: { fromDaemon: true, response: resp },
							};
						}
						return toolResultFromDaemon(resp);
					},
					async () => {
						if (!resolved) throw new Error("rbtr CLI not available");
						const result = await runRbtr(pi, resolved, ["read-symbol", params.symbol], { signal, timeout: 30_000 });
						const text = result.stdout.trim();
						if (!text) {
							return {
								content: [{ type: "text", text: `Symbol not found: ${params.symbol}` }],
								details: { fromCli: true, symbol: params.symbol, found: false },
							};
						}
						return toolResultFromCli(text, { symbol: params.symbol, found: true });
					},
				);
			} catch (err) {
				return mapDaemonError(err);
			}
		},
	});

	pi.registerTool({
		name: "rbtr_find_refs",
		label: "rbtr find-refs",
		description: "Find references to a symbol via the dependency graph (imports, tests, docs).",
		promptSnippet: "Find references to a symbol via the dependency graph (imports, tests, docs)",
		promptGuidelines: [
			"Use rbtr_find_refs to find where a symbol is imported, tested, or documented.",
			"Shows structural relationships, not just text matches.",
		],
		parameters: Type.Object({
			symbol: Type.String({ description: "Symbol name" }),
		}),
		renderCall: (args, theme) => renderFindRefsCall(args, theme),
		renderResult: (result, options, theme) => renderFindRefsResult(result, options, theme),

		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			try {
				return await withFallback<ToolReturn>(
					async () => {
						const resp = await session.send({
							kind: "find_refs",
							repo: ctx.cwd,
							symbol: params.symbol,
						});
						if (resp.edges.length === 0) {
							return {
								content: [{ type: "text", text: `No references found for: ${params.symbol}` }],
								details: { fromDaemon: true, response: resp },
							};
						}
						return toolResultFromDaemon(resp);
					},
					async () => {
						if (!resolved) throw new Error("rbtr CLI not available");
						const result = await runRbtr(pi, resolved, ["find-refs", params.symbol], { signal, timeout: 30_000 });
						const text = result.stdout.trim();
						if (!text) {
							return {
								content: [{ type: "text", text: `No references found for: ${params.symbol}` }],
								details: { fromCli: true, symbol: params.symbol, found: false },
							};
						}
						return toolResultFromCli(text, { symbol: params.symbol, found: true });
					},
				);
			} catch (err) {
				return mapDaemonError(err);
			}
		},
	});

	pi.registerTool({
		name: "rbtr_changed_symbols",
		label: "rbtr changed-symbols",
		description: "Show symbols that changed between two git refs (structural diff).",
		promptSnippet: "Show symbols that changed between two git refs (structural diff)",
		promptGuidelines: [
			"Use rbtr_changed_symbols to understand what code changed structurally between branches.",
			"Shows function/class-level changes, not line-level diffs.",
		],
		parameters: Type.Object({
			base: Type.String({ description: "Base ref (e.g. main)" }),
			head: Type.String({ description: "Head ref (e.g. feature-branch)" }),
		}),
		renderCall: (args, theme) => renderChangedSymbolsCall(args, theme),
		renderResult: (result, options, theme) => renderChangedSymbolsResult(result, options, theme),

		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			try {
				return await withFallback<ToolReturn>(
					async () => {
						const resp = await session.send({
							kind: "changed_symbols",
							repo: ctx.cwd,
							base: params.base,
							head: params.head,
						});
						if (resp.chunks.length === 0) {
							return {
								content: [{ type: "text", text: `No changed symbols between ${params.base} and ${params.head}` }],
								details: { fromDaemon: true, response: resp },
							};
						}
						return toolResultFromDaemon(resp);
					},
					async () => {
						if (!resolved) throw new Error("rbtr CLI not available");
						const result = await runRbtr(
							pi,
							resolved,
							["changed-symbols", "--base", params.base, "--head", params.head],
							{ signal, timeout: 30_000 },
						);
						const text = result.stdout.trim();
						if (!text) {
							return {
								content: [{ type: "text", text: `No changed symbols between ${params.base} and ${params.head}` }],
								details: { fromCli: true, base: params.base, head: params.head, found: false },
							};
						}
						return toolResultFromCli(text, { base: params.base, head: params.head, found: true });
					},
				);
			} catch (err) {
				return mapDaemonError(err);
			}
		},
	});

	pi.registerTool({
		name: "rbtr_list_symbols",
		label: "rbtr list-symbols",
		description: "List all symbols (functions, classes, methods) in a file as a structural table of contents.",
		promptSnippet: "List all symbols (functions, classes, methods) in a file as a structural table of contents",
		promptGuidelines: [
			"Use rbtr_list_symbols to understand the structure of a file before reading specific parts.",
			"More informative than reading the whole file.",
		],
		parameters: Type.Object({
			file: Type.String({ description: "File path (relative to repo root)" }),
		}),
		renderCall: (args, theme) => renderListSymbolsCall(args, theme),
		renderResult: (result, options, theme) => renderListSymbolsResult(result, options, theme),

		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			try {
				return await withFallback<ToolReturn>(
					async () => {
						const resp = await session.send({
							kind: "list_symbols",
							repo: ctx.cwd,
							file_path: params.file,
						});
						if (resp.chunks.length === 0) {
							return {
								content: [{ type: "text", text: `No symbols found in: ${params.file}` }],
								details: { fromDaemon: true, response: resp },
							};
						}
						return toolResultFromDaemon(resp);
					},
					async () => {
						if (!resolved) throw new Error("rbtr CLI not available");
						const result = await runRbtr(pi, resolved, ["list-symbols", params.file], {
							signal,
							timeout: 30_000,
						});
						const text = result.stdout.trim();
						if (!text) {
							return {
								content: [{ type: "text", text: `No symbols found in: ${params.file}` }],
								details: { fromCli: true, file: params.file, found: false },
							};
						}
						return toolResultFromCli(text, { file: params.file, found: true });
					},
				);
			} catch (err) {
				return mapDaemonError(err);
			}
		},
	});
}
