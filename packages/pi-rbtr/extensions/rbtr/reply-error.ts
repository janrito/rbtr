/**
 * An rbtr `ErrorResponse`, thrown where a reply was expected.
 */

import type { ErrorCode, ErrorResponse } from "./generated/protocol.js";

/**
 * Thrown when rbtr answers with an ``ErrorResponse``, from the daemon
 * or from the CLI's JSON output.
 *
 * Carrying the typed ``code`` lets callers branch on it without
 * string-matching the message.
 */
export class RbtrReplyError extends Error {
  readonly code: ErrorCode;

  constructor(response: ErrorResponse) {
    super(response.message);
    this.name = "RbtrReplyError";
    this.code = response.code;
  }
}
