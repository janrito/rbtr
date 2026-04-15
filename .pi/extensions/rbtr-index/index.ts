/**
 * rbtr-index — pi extension for the rbtr structural code index.
 *
 * Gives the LLM access to rbtr's code index via registered tools.
 * Shells out to the `rbtr` CLI for all operations.
 *
 * Placement: .pi/extensions/rbtr-index/index.ts
 */

import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { getSettingsListTheme } from "@mariozechner/pi-coding-agent";
import { Container, type SettingItem, SettingsList } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";
import { resolveCommand, runRbtr, runRbtrJson, type ResolvedCommand } from "./exec.js";
import {
	renderBuildCall,
	renderBuildResult,
	renderChangedSymbolsCall,
	renderChangedSymbolsResult,
	renderFindRefsCall,
	renderFindRefsResult,
	renderListSymbolsCall,
	renderListSymbolsResult,
	renderReadSymbolCall,
	renderReadSymbolResult,
	renderSearchCall,
	renderSearchResult,
	renderStatusCall,
	renderStatusResult,
} from "./render.js";
import { loadSettings, saveProjectSettings, type RbtrIndexSettings } from "./settings.js";

interface IndexStatus {
	exists: boolean;
	db_path?: string;
	total_chunks?: number;
}

interface BuildStats {
	total_chunks: number;
	total_edges: number;
	total_files: number;
	skipped_files: number;
	parsed_files: number;
	elapsed_seconds: number;
}

interface BuildResult {
	ref: string;
	stats: BuildStats;
	errors: string[];
}

export default function rbtrIndexExtension(pi: ExtensionAPI) {
	let resolved: ResolvedCommand | null = null;
	let settings: RbtrIndexSettings = { command: "rbtr", autoBuild: true };
	let cliAvailable = false;
	let buildPromise: Promise<BuildResult> | null = null;

	async function checkStatus(): Promise<IndexStatus | null> {
		if (!resolved) return null;
		try {
			const results = await runRbtrJson<IndexStatus>(pi, resolved, ["status"], { timeout: 5000 });
			return results[0] ?? null;
		} catch {
			return null;
		}
	}

	// ── Build management ───────────────────────────────────────

	function isBuilding(): boolean {
		return buildPromise !== null;
	}

	/**
	 * Launch a background index build. Returns false if a build
	 * is already running. Updates footer status and notifies
	 * on completion.
	 */
	function startBuild(ctx: ExtensionContext, ref = "HEAD"): boolean {
		if (!resolved || !cliAvailable) return false;
		if (buildPromise) return false;

		ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("warning", "rbtr: building\u2026"));

		const captured = { setStatus: ctx.ui.setStatus.bind(ctx.ui), notify: ctx.ui.notify.bind(ctx.ui), theme: ctx.ui.theme };

		buildPromise = runRbtrJson<BuildResult>(pi, resolved, ["build", "--ref", ref], { timeout: 600_000 })
			.then(async (results) => {
				const result = results[0];
				if (!result) {
					// Build produced no JSON output — re-check status for the count
					const status = await checkStatus();
					const count = status?.total_chunks ?? 0;
					captured.setStatus("rbtr", captured.theme.fg("success", `rbtr: ${count} symbols`));
					captured.notify("Index build completed.", "info");
					return { ref, stats: { total_chunks: count, total_edges: 0, total_files: 0, skipped_files: 0, parsed_files: 0, elapsed_seconds: 0 }, errors: [] };
				}
				const s = result.stats;
				captured.setStatus("rbtr", captured.theme.fg("success", `rbtr: ${s.total_chunks} symbols`));
				captured.notify(
					`Index built: ${s.total_chunks} chunks, ${s.total_edges} edges \u2014 ${s.elapsed_seconds.toFixed(1)}s` +
						(result.errors.length > 0 ? `\n${result.errors.length} error(s)` : ""),
					result.errors.length > 0 ? "warning" : "info",
				);
				return result;
			})
			.catch((err) => {
				captured.setStatus("rbtr", captured.theme.fg("error", "rbtr: build failed"));
				captured.notify(`Index build failed: ${err instanceof Error ? err.message : String(err)}`, "error");
				throw err;
			})
			.finally(() => {
				buildPromise = null;
			});

		return true;
	}

	/**
	 * Guard used by query tools: throws if CLI unavailable or
	 * build is in progress so the LLM gets a clear error.
	 */
	function requireReady(): void {
		if (!resolved || !cliAvailable) {
			throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
		}
		if (isBuilding()) {
			throw new Error("Index build is in progress. Try again after it completes.");
		}
	}

	// ── Session lifecycle ───────────────────────────────────────

	pi.on("session_start", async (_event, ctx) => {
		settings = loadSettings(ctx.cwd);
		resolved = resolveCommand(settings.command);

		const status = await checkStatus();
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
		} else if (settings.autoBuild) {
			startBuild(ctx);
		} else {
			ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("warning", "rbtr: no index \u2014 /rbtr-build"));
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

	// ── Commands ────────────────────────────────────────────────

	pi.registerCommand("rbtr-status", {
		description: "Show rbtr index status",
		handler: async (_args, ctx) => {
			if (!cliAvailable || !resolved) {
				ctx.ui.notify("rbtr CLI not available", "error");
				return;
			}
			const status = await checkStatus();
			if (!status) {
				ctx.ui.notify("Failed to get index status", "error");
				return;
			}
			if (status.exists) {
				ctx.ui.notify(`Index: ${status.total_chunks} symbols\nPath: ${status.db_path}`, "info");
			} else {
				ctx.ui.notify("No index found. Use /rbtr-build to create one.", "warning");
			}
		},
	});

	pi.registerCommand("rbtr-build", {
		description: "Build or rebuild the rbtr code index",
		handler: async (_args, ctx) => {
			if (!cliAvailable || !resolved) {
				ctx.ui.notify("rbtr CLI not available", "error");
				return;
			}
			if (isBuilding()) {
				ctx.ui.notify("Build already in progress", "warning");
				return;
			}
			startBuild(ctx);
			ctx.ui.notify("Index build started. Progress in the footer.", "info");
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
						id: "autoBuild",
						label: "Auto-build on session start",
						currentValue: settings.autoBuild ? "on" : "off",
						values: ["on", "off"],
					},
				];

				const container = new Container();

				// Header with resolved command info
				container.addChild({
					render(_width: number) {
						const desc = resolved ? resolved.description : "not resolved";
						return [
							theme.fg("accent", theme.bold("rbtr Index Settings")),
							"",
							`${theme.fg("muted", "Command:")} ${settings.command}`,
							`${theme.fg("muted", "Resolved:")} ${theme.fg("dim", desc)}`,
							`${theme.fg("muted", "CLI available:")} ${cliAvailable ? theme.fg("success", "yes") : theme.fg("error", "no")}`,
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
						if (id === "autoBuild") {
							settings.autoBuild = newValue === "on";
							saveProjectSettings(ctx.cwd, { autoBuild: settings.autoBuild });
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
		name: "rbtr_build",
		label: "rbtr build",
		description: "Build or update the structural code index for the repository.",
		promptSnippet: "Build or update the structural code index for the repository",
		promptGuidelines: [
			"Use rbtr_build when the user asks to index the codebase or when rbtr_search returns no results.",
			"Building is incremental — unchanged files are skipped.",
			"The build runs in background. Use rbtr_status to check when it completes.",
		],
		parameters: Type.Object({
			ref: Type.Optional(Type.String({ description: "Git ref to index (default: HEAD)" })),
		}),
		renderCall: (args, theme) => renderBuildCall(args, theme),
		renderResult: (result, options, theme) => renderBuildResult(result, options, theme),

		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			if (!resolved || !cliAvailable) {
				throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
			}
			if (isBuilding()) {
				return {
					content: [{ type: "text", text: "Build already in progress. Use rbtr_status to check progress." }],
					details: { status: "in_progress" },
				};
			}

			const ref = params.ref || "HEAD";
			startBuild(ctx, ref);

			return {
				content: [
					{
						type: "text",
						text: `Build started for ref ${ref}. Progress is shown in the footer. Use rbtr_status to check when complete.`,
					},
				],
				details: { status: "started", ref },
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

		async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
			if (!resolved || !cliAvailable) {
				throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
			}
			if (isBuilding()) {
				return {
					content: [{ type: "text", text: "Index build is in progress. Check again shortly." }],
					details: { building: true },
				};
			}

			const status = await checkStatus();
			if (!status) {
				throw new Error("Failed to check index status");
			}

			if (status.exists) {
				return {
					content: [{ type: "text", text: `Index: ${status.total_chunks} symbols\nPath: ${status.db_path}` }],
					details: status,
				};
			}

			return {
				content: [{ type: "text", text: "No index found. Use rbtr_build to create one." }],
				details: status,
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

		async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
			requireReady();

			const args = ["search", params.query];
			if (params.limit) args.push("--limit", String(params.limit));

			const result = await runRbtr(pi, resolved!, args, { signal, timeout: 30_000 });
			const text = result.stdout.trim();

			if (!text) {
				return {
					content: [{ type: "text", text: "No results found." }],
					details: { results: [] },
				};
			}

			return {
				content: [{ type: "text", text }],
				details: { query: params.query, limit: params.limit },
			};
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

		async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
			requireReady();

			const result = await runRbtr(pi, resolved!, ["read-symbol", params.symbol], { signal, timeout: 30_000 });
			const text = result.stdout.trim();

			if (!text) {
				return {
					content: [{ type: "text", text: `Symbol not found: ${params.symbol}` }],
					details: { symbol: params.symbol, found: false },
				};
			}

			return {
				content: [{ type: "text", text }],
				details: { symbol: params.symbol, found: true },
			};
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

		async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
			requireReady();

			const result = await runRbtr(pi, resolved!, ["find-refs", params.symbol], { signal, timeout: 30_000 });
			const text = result.stdout.trim();

			if (!text) {
				return {
					content: [{ type: "text", text: `No references found for: ${params.symbol}` }],
					details: { symbol: params.symbol, found: false },
				};
			}

			return {
				content: [{ type: "text", text }],
				details: { symbol: params.symbol, found: true },
			};
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

		async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
			requireReady();

			const result = await runRbtr(pi, resolved!, ["changed-symbols", "--base", params.base, "--head", params.head], {
				signal,
				timeout: 30_000,
			});
			const text = result.stdout.trim();

			if (!text) {
				return {
					content: [{ type: "text", text: `No changed symbols between ${params.base} and ${params.head}` }],
					details: { base: params.base, head: params.head, found: false },
				};
			}

			return {
				content: [{ type: "text", text }],
				details: { base: params.base, head: params.head, found: true },
			};
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

		async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
			requireReady();

			const result = await runRbtr(pi, resolved!, ["list-symbols", params.file], { signal, timeout: 30_000 });
			const text = result.stdout.trim();

			if (!text) {
				return {
					content: [{ type: "text", text: `No symbols found in: ${params.file}` }],
					details: { file: params.file, found: false },
				};
			}

			return {
				content: [{ type: "text", text }],
				details: { file: params.file, found: true },
			};
		},
	});
}
