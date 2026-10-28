/**
 * An rbtr `ErrorResponse`, thrown where a reply was expected.
 */

import type { ErrorCode, ErrorResponse } from "./generated/protocol.js";

/**
 * Thrown when rbtr answers with an ``ErrorResponse``, from the daemon
 * or from the CLI's JSON output.
 *
 * Carrying the typed ``code`` lets callers branch on it without
 * string-matching the message.  Running out of memory adds how to raise
 * the limit, which DuckDB's own message does not name.
 */
export class RbtrReplyError extends Error {
  readonly code: ErrorCode;

  constructor(response: ErrorResponse) {
    super(
      response.code === "out_of_memory"
        ? `${response.message}\nRaise duckdb_memory_limit in config.toml or with RBTR_DUCKDB_MEMORY_LIMIT.`
        : response.message,
    );
    this.name = "RbtrReplyError";
    this.code = response.code;
  }
}
