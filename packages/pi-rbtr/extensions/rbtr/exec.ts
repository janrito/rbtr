/**
 * CLI invocation and command resolution for rbtr.
 *
 * Resolves the `command` setting into an executable + base args pair,
 * then provides helpers to run rbtr subcommands and parse NDJSON output.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export interface ResolvedCommand {
	executable: string;
	baseArgs: string[];
	description: string;
}

/**
 * Resolve a `command` setting string into an executable and base args.
 *
 * Supported formats:
 * - `"rbtr"` — direct invocation (uv tool install)
 * - `"uvx"` — one-shot from published package
 * - `"uvx --from <path>"` — one-shot from local directory
 */
export function resolveCommand(command: string): ResolvedCommand {
	const trimmed = command.trim();

	if (trimmed === "rbtr") {
		return { executable: "rbtr", baseArgs: ["--json"], description: "rbtr (PATH)" };
	}

	if (trimmed === "uvx") {
		return { executable: "uvx", baseArgs: ["rbtr", "--json"], description: "uvx rbtr" };
	}

	const uvxFromMatch = trimmed.match(/^uvx\s+--from\s+(.+)$/);
	if (uvxFromMatch) {
		const fromPath = uvxFromMatch[1].trim();
		return {
			executable: "uvx",
			baseArgs: ["--from", fromPath, "rbtr", "--json"],
			description: `uvx --from ${fromPath}`,
		};
	}

	// Fallback: treat as direct command name
	return { executable: trimmed, baseArgs: ["--json"], description: `${trimmed} (PATH)` };
}

/**
 * Parse NDJSON (newline-delimited JSON) output into typed objects.
 */
export function parseNdjson<T>(stdout: string): T[] {
	return stdout
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line) as T);
}

export interface RbtrExecResult {
	stdout: string;
	stderr: string;
	code: number | null;
}

/**
 * Run an rbtr subcommand and return raw output.
 * Throws on non-zero exit code or if the command is not found.
 */
export async function runRbtr(
	pi: ExtensionAPI,
	resolved: ResolvedCommand,
	args: string[],
	options?: { signal?: AbortSignal; timeout?: number },
): Promise<RbtrExecResult> {
	let result: { stdout: string; stderr: string; code: number | null; killed: boolean };
	try {
		result = await pi.exec(resolved.executable, [...resolved.baseArgs, ...args], options);
	} catch (err) {
		throw new Error(`Failed to run rbtr: ${err instanceof Error ? err.message : String(err)}`);
	}

	if (result.killed) {
		throw new Error("rbtr was killed (timeout or signal)");
	}

	if (result.code !== 0) {
		const msg = result.stderr?.trim() || `rbtr exited with code ${result.code}`;
		throw new Error(msg);
	}

	return { stdout: result.stdout, stderr: result.stderr, code: result.code };
}

/**
 * Run an rbtr subcommand and parse NDJSON output.
 */
export async function runRbtrJson<T>(
	pi: ExtensionAPI,
	resolved: ResolvedCommand,
	args: string[],
	options?: { signal?: AbortSignal; timeout?: number },
): Promise<T[]> {
	const result = await runRbtr(pi, resolved, args, options);
	return parseNdjson<T>(result.stdout);
}
