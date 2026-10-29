/**
 * CLI invocation and command resolution for rbtr.
 *
 * Resolves the `command` setting into an executable + base args pair,
 * then provides helpers to run rbtr subcommands and parse their JSON
 * response object.
 *
 * When the command is `"rbtr"`, resolution follows a cascade:
 * 1. rbtr in the active Python environment (`python3 -m rbtr`)
 * 2. rbtr on PATH
 * 3. uvx rbtr
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import type { DaemonStatusReport, ErrorResponse } from "./generated/protocol.js";
import { RbtrReplyError } from "./reply-error.js";

export interface ResolvedCommand {
  executable: string;
  baseArgs: string[];
  description: string;
}

/**
 * The rbtr a session runs and what its `daemon status` reported, or the
 * last candidate tried and why its `daemon status` did not answer.
 */
export type FoundRbtr =
  | { kind: "found"; command: ResolvedCommand; daemon: DaemonStatusReport }
  | { kind: "failed"; command: ResolvedCommand; reason: string };

const UVX_RBTR: ResolvedCommand = { executable: "uvx", baseArgs: ["rbtr", "--json"], description: "uvx rbtr" };

/**
 * Resolve a `command` setting string into an executable and base args,
 * and ask that rbtr for its daemon's status.
 *
 * Supported formats:
 * - `"rbtr"` — cascade: `python -m rbtr` → PATH `rbtr` → `uvx rbtr`;
 *   the first candidate whose `daemon status` answers is taken
 * - `"uvx"` — one-shot from published package
 * - `"uvx --from <path>"` — one-shot from local directory
 */
export async function resolveCommand(pi: Pick<ExtensionAPI, "exec">, command: string): Promise<FoundRbtr> {
  const trimmed = command.trim();
  // `daemon status` answers whenever rbtr runs, daemon or not. pi reports a
  // program that cannot start as a silent exit 1, so a failure says only
  // that this candidate does not run.
  const daemonStatus = async (candidate: ResolvedCommand, timeout?: number): Promise<FoundRbtr> => {
    const failed = (reason: string): FoundRbtr => ({ kind: "failed", command: candidate, reason });
    try {
      const result = await pi.exec(candidate.executable, [...candidate.baseArgs, "daemon", "status"], { timeout });
      if (result.killed) return failed("no answer in time");
      if (result.code !== 0) return failed(result.stderr.trim().split("\n").at(-1) || `exit ${result.code}`);
      const daemon = statusReport(result.stdout);
      return daemon === null
        ? failed("unreadable `daemon status` reply")
        : { kind: "found", command: candidate, daemon };
    } catch (err) {
      return failed(err instanceof Error ? err.message : String(err));
    }
  };

  if (trimmed === "rbtr") {
    const candidates: ResolvedCommand[] = [
      { executable: "python3", baseArgs: ["-m", "rbtr", "--json"], description: "python -m rbtr" },
      { executable: "rbtr", baseArgs: ["--json"], description: "rbtr (PATH)" },
    ];
    for (const candidate of candidates) {
      const found = await daemonStatus(candidate, 5000);
      if (found.kind === "found") return found;
    }
    return daemonStatus(UVX_RBTR);
  }

  return daemonStatus(parseCommand(trimmed));
}

/** The `uvx`, `uvx --from <path>` or plain-executable form of a `command` setting. */
function parseCommand(command: string): ResolvedCommand {
  if (command === "uvx") return UVX_RBTR;

  const uvxFromMatch = command.match(/^uvx\s+--from\s+(.+)$/);
  if (uvxFromMatch) {
    const fromPath = uvxFromMatch[1].trim();
    return {
      executable: "uvx",
      baseArgs: ["--from", fromPath, "rbtr", "--json"],
      description: `uvx --from ${fromPath}`,
    };
  }

  // Fallback: treat as direct command name
  return { executable: command, baseArgs: ["--json"], description: `${command} (PATH)` };
}

export interface RbtrExecResult {
  stdout: string;
  stderr: string;
  code: number | null;
}

/**
 * The `ErrorResponse` a failed `rbtr --json` command printed on stdout,
 * or `null` when it printed none (it crashed, or never ran).
 */
export function errorReply(stdout: string): ErrorResponse | null {
  try {
    const parsed: unknown = JSON.parse(stdout);
    return isErrorResponse(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * The `DaemonStatusReport` an `rbtr --json daemon status` or
 * `daemon start` printed, or `null` when it printed none (an older
 * rbtr's `daemon start` prints `{"kind":"ok"}`).
 */
export function statusReport(stdout: string): DaemonStatusReport | null {
  try {
    const parsed: unknown = JSON.parse(stdout);
    return isStatusReport(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function isStatusReport(value: unknown): value is DaemonStatusReport {
  return typeof value === "object" && value !== null && "running" in value && typeof value.running === "boolean";
}

function isErrorResponse(value: unknown): value is ErrorResponse {
  return typeof value === "object" && value !== null && "kind" in value && value.kind === "error";
}

/**
 * Run an rbtr subcommand and return raw output.
 * Throws on non-zero exit code or if the command is not found: an
 * `RbtrReplyError` when rbtr printed its `ErrorResponse`, otherwise an
 * error carrying stderr.
 */
export async function runRbtr(
  pi: Pick<ExtensionAPI, "exec">,
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
    const reply = errorReply(result.stdout);
    if (reply) throw new RbtrReplyError(reply);
    const msg = result.stderr?.trim() || `rbtr exited with code ${result.code}`;
    throw new Error(msg);
  }

  return { stdout: result.stdout, stderr: result.stderr, code: result.code };
}

/**
 * The rbtr commands run so far, chained. A command run with no daemon
 * opens the index itself and holds it alone while it runs, so two at
 * once fail with `index_locked`; each starts when the one before it ends.
 */
let previousCommand: Promise<unknown> = Promise.resolve();

/**
 * Run an rbtr subcommand and parse its single JSON response object.
 * Commands run one at a time (`previousCommand`); one aborted while
 * waiting does not start.
 */
export async function runRbtrJson<T>(
  pi: ExtensionAPI,
  resolved: ResolvedCommand,
  args: string[],
  options?: { signal?: AbortSignal; timeout?: number },
): Promise<T> {
  const command = previousCommand.then(async () => {
    options?.signal?.throwIfAborted();
    const result = await runRbtr(pi, resolved, args, options);
    return JSON.parse(result.stdout) as T;
  });
  previousCommand = command.catch(() => undefined);
  return command;
}
