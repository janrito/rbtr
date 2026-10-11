/**
 * ZMQ REQ client for the rbtr daemon.
 *
 * Python models are the source of truth.  Every request/response
 * type on the wire is imported from the generated protocol
 * module — if it is not in there, it does not exist.
 *
 * Endpoint discovery runs `rbtr --json daemon status` with the
 * session's resolved command (the CLI knows how its own runtime dir
 * resolves from its config / `RBTR_*_DIR` overrides).  The status
 * file path is **not** read directly.
 *
 * One REQ socket per call; no pooling.  ZMQ REQ enforces strict
 * send/recv alternation, and the CLI is not worth the complexity
 * of a long-lived socket with recovery logic.
 */

import { spawn } from "node:child_process";
import { Request as ZmqRequest } from "zeromq";

import type { ResolvedCommand } from "./exec.js";
import type { DaemonStatusReport, Request, Response } from "./generated/protocol.js";
import { RbtrReplyError } from "./reply-error.js";

/**
 * Returned when ``rbtr daemon status --json`` succeeds.
 *
 * The shape is the generated ``DaemonStatusReport``.
 */
export type DaemonStatus = DaemonStatusReport;

export interface SendOptions {
  /** Milliseconds to wait for the daemon's reply.  Default 120 000. */
  waitBudgetMs?: number;
  /** Milliseconds to wait for the outbound send.  Default 5 000. */
  sendTimeout?: number;
  /** Stops the wait: the socket is closed and the call rejects with the abort reason. */
  signal?: AbortSignal;
}

/** The reply a request of *kind* K gets: ``Response`` narrowed by kind. */
export type ResponseFor<K extends string> = Extract<Response, { kind: K }>;

/**
 * Ask the CLI, run as *resolved*, for the running daemon's status.
 *
 * The path to the status file depends on the user's runtime dir
 * (resolved from config / `RBTR_*_DIR` overrides), so the CLI is
 * the source of truth — we do not read the JSON file directly.
 */
export async function queryDaemonStatus(resolved: ResolvedCommand): Promise<DaemonStatus> {
  return new Promise<DaemonStatus>((resolve, reject) => {
    const proc = spawn(resolved.executable, [...resolved.baseArgs, "daemon", "status"], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    proc.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`rbtr daemon status exited with code ${code}: ${stderr.trim()}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout) as DaemonStatus);
      } catch (err) {
        reject(new Error(`rbtr daemon status produced invalid JSON: ${err instanceof Error ? err.message : err}`));
      }
    });
  });
}

/**
 * Send one request to the daemon at *rpcEndpoint* (an ``ipc://``
 * URI) and return the matching response.
 *
 * Throws ``RbtrReplyError`` when the daemon replies with an
 * ``ErrorResponse`` — on success the returned type is narrowed
 * to the response matching the request's ``kind``.
 *
 * The request is sent once and the reply waited for.  A daemon
 * that is indexing serves the request late rather than not at
 * all, and sending a second copy only queues more work behind
 * the first.  Silence past the budget is reported to the caller,
 * which falls back to the CLI.  An abort of *signal* closes the
 * socket and rejects with the signal's reason.
 */
export async function send<R extends Request>(
  request: R,
  options: SendOptions & { rpcEndpoint: string },
): Promise<ResponseFor<R["kind"]>> {
  const { rpcEndpoint: endpoint, signal } = options;
  signal?.throwIfAborted();
  const sock = new ZmqRequest({
    receiveTimeout: options.waitBudgetMs ?? 120_000,
    sendTimeout: options.sendTimeout ?? 5_000,
  });
  // Closing the socket ends a pending send or receive.
  const onAbort = () => sock.close();
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    sock.connect(endpoint);
    await sock.send(JSON.stringify(request));
    const [reply] = await sock.receive();
    const response = JSON.parse(reply.toString()) as Response;

    if (response.kind === "error") {
      throw new RbtrReplyError(response);
    }

    // Discriminator guarantees the kind matches the request's
    // kind for every non-error response (the server is typed).
    return response as ResponseFor<R["kind"]>;
  } catch (err) {
    if (signal?.aborted) throw signal.reason;
    throw err;
  } finally {
    signal?.removeEventListener("abort", onAbort);
    if (!sock.closed) sock.close();
  }
}
