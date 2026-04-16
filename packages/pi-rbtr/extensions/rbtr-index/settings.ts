/**
 * Settings for the rbtr-index extension.
 *
 * Config files (project overrides global):
 * - ~/.pi/agent/rbtr-index.json
 * - <cwd>/.pi/rbtr-index.json
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { getAgentDir } from "@mariozechner/pi-coding-agent";

export interface RbtrIndexSettings {
	command: string;
	autoIndex: boolean;
}

const DEFAULTS: RbtrIndexSettings = {
	command: "rbtr",
	autoIndex: true,
};

const CONFIG_FILENAME = "rbtr-index.json";

function loadJsonFile(path: string): Partial<RbtrIndexSettings> {
	if (!existsSync(path)) return {};
	try {
		return JSON.parse(readFileSync(path, "utf-8"));
	} catch {
		return {};
	}
}

export function loadSettings(cwd: string): RbtrIndexSettings {
	const globalPath = join(getAgentDir(), CONFIG_FILENAME);
	const projectPath = join(cwd, ".pi", CONFIG_FILENAME);

	const global = loadJsonFile(globalPath);
	const project = loadJsonFile(projectPath);

	return { ...DEFAULTS, ...global, ...project };
}

export function saveProjectSettings(cwd: string, settings: Partial<RbtrIndexSettings>): void {
	const projectDir = join(cwd, ".pi");
	const projectPath = join(projectDir, CONFIG_FILENAME);

	const existing = loadJsonFile(projectPath);
	const merged = { ...existing, ...settings };

	mkdirSync(projectDir, { recursive: true });
	writeFileSync(projectPath, `${JSON.stringify(merged, null, 2)}\n`, "utf-8");
}
