/**
 * rbtr-index — pi extension for the rbtr structural code index.
 *
 * Gives the LLM access to rbtr's code index via registered tools.
 * Shells out to the `rbtr` CLI for all operations.
 *
 * Placement: .pi/extensions/rbtr-index/index.ts
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { getSettingsListTheme } from "@mariozechner/pi-coding-agent";
import { Container, type SettingItem, SettingsList } from "@mariozechner/pi-tui";
import { resolveCommand, runRbtrJson, type ResolvedCommand } from "./exec.js";
import { loadSettings, saveProjectSettings, type RbtrIndexSettings } from "./settings.js";

interface IndexStatus {
	exists: boolean;
	db_path?: string;
	total_chunks?: number;
}

export default function rbtrIndexExtension(pi: ExtensionAPI) {
	let resolved: ResolvedCommand | null = null;
	let settings: RbtrIndexSettings = { command: "rbtr", autoBuild: true };
	let cliAvailable = false;

	async function checkStatus(): Promise<IndexStatus | null> {
		if (!resolved) return null;
		try {
			const results = await runRbtrJson<IndexStatus>(pi, resolved, ["status"], { timeout: 5000 });
			return results[0] ?? null;
		} catch {
			return null;
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
		} else {
			ctx.ui.setStatus("rbtr", ctx.ui.theme.fg("warning", "rbtr: no index — /rbtr-build"));
		}
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
}
