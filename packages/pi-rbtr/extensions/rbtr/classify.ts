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
 * rbtr has already answered `daemon status`, so it runs: a start
 * that fails is a locked index or a transient failure.
 */
export function classifyDaemonFailure(code: number | null, reply: ErrorResponse | null, stderr: string): DaemonFailure {
  if (reply?.code === "index_locked") {
    return { kind: "db-locked", message: "rbtr index temporarily unavailable (database busy); will retry." };
  }
  const message = reply?.message ?? stderr.trim();
  return { kind: "transient", message: message || `rbtr daemon command failed (exit ${code ?? "?"}).` };
}

export type StartupDecision =
  | { kind: "indexed" } // the repo has indexed refs: show the footer
  | { kind: "empty"; index: boolean } // no index, or not this repo: index if enabled
  | { kind: "transient"; index: boolean }; // daemon/db busy: warn, index if enabled

/**
 * Decide what `session_start` should do, once rbtr runs, from its index
 * status: `null` when the status could not be read. A transient failure
 * still attempts indexing so the repo gets indexed once the daemon is
 * healthy.
 */
export function decideStartupDecision(status: StatusResponse | null, autoIndex: boolean): StartupDecision {
  if (status === null) return { kind: "transient", index: autoIndex };
  if (status.state === "indexed" || status.state === "ref_not_indexed") return { kind: "indexed" };
  return { kind: "empty", index: autoIndex };
}
