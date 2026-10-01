/**
 * Pure classification of daemon failures and session-start
 * status outcomes.
 *
 * No pi / network / filesystem dependency, so the decision logic
 * is unit-testable in isolation.  The session handler in
 * `index.ts` performs the side effects (notify, footer, index)
 * driven by these verdicts.
 */

import type { ErrorResponse, StatusResponse } from "./generated/protocol.js";

export type DaemonFailureKind =
  | "missing-cli" // the rbtr command itself could not be run
  | "db-locked" // the index DB is held by another process
  | "transient"; // any other start/restart failure — retryable

export interface DaemonFailure {
  kind: DaemonFailureKind;
  message: string;
}

/**
 * Classify a failed `rbtr daemon …` invocation into an actionable
 * category and message, from its exit code, the `ErrorResponse`
 * rbtr printed (`null` when it printed none), and its stderr.
 *
 * A command that never ran leaves only stderr and its exit code.
 * Only `missing-cli` should disable rbtr for the session; the
 * others are transient and must not be reported as a missing
 * CLI.
 */
export function classifyDaemonFailure(code: number | null, reply: ErrorResponse | null, stderr: string): DaemonFailure {
  const text = stderr.toLowerCase();
  if (code === 127 || text.includes("command not found") || text.includes("no such file") || text.includes("enoent")) {
    return { kind: "missing-cli", message: "rbtr CLI not found. Install with: uv tool install rbtr" };
  }
  if (reply?.code === "index_locked") {
    return { kind: "db-locked", message: "rbtr index temporarily unavailable (database busy); will retry." };
  }
  const message = reply?.message ?? stderr.trim();
  return { kind: "transient", message: message || `rbtr daemon command failed (exit ${code ?? "?"}).` };
}

export type StartupDecision =
  | { kind: "missing-cli" } // CLI genuinely absent: warn, do not index
  | { kind: "indexed" } // status has refs: show the footer
  | { kind: "empty"; index: boolean } // no refs yet: index if enabled
  | { kind: "transient"; index: boolean }; // daemon/db busy: warn, index if enabled

/**
 * Decide what `session_start` should do after querying status.
 *
 * A `null` status is ambiguous: it means either the CLI is
 * genuinely unresolvable, or a transient daemon/lock failure.
 * `cliResolved` (did the command resolve on PATH at all?)
 * disambiguates — only a truly unresolved command is treated as
 * `missing-cli`.  A transient failure still attempts indexing so
 * the repo gets indexed once the daemon is healthy.
 */
export function decideStartupDecision(
  cliResolved: boolean,
  status: StatusResponse | null,
  autoIndex: boolean,
): StartupDecision {
  if (status === null) {
    if (!cliResolved) return { kind: "missing-cli" };
    return { kind: "transient", index: autoIndex };
  }
  const indexed = (status.indexed_refs ?? []).length > 0;
  if (indexed) return { kind: "indexed" };
  return { kind: "empty", index: autoIndex };
}
