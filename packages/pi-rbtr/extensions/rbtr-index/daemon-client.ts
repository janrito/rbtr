/**
 * ZMQ REQ client for the rbtr daemon.
 *
 * Python models are the source of truth.  Every request/response
 * type on the wire is imported from the generated protocol
 * module — if it is not in there, it does not exist.
 *
 * Endpoint discovery runs `rbtr daemon status --json` (the CLI
 * knows how its own `RBTR_HOME` setting resolves).  The status
 * file path is **not** read directly.
 *
 * One REQ socket per call; no pooling.  ZMQ REQ enforces strict
 * send/recv alternation, and the CLI is not worth the complexity
 * of a long-lived socket with recovery logic.
 */

import { spawn } from "node:child_process";
import { Request as ZmqRequest } from "zeromq";

import type { DaemonStatusReport, ErrorCode, ErrorResponse, Request, Response } from "./generated/protocol.js";

/**
 * Thrown when the daemon answers with an ``ErrorResponse``.
 *
 * Carrying the typed ``code`` lets callers branch on it without
 * string-matching the message.
 */
export class RbtrDaemonError extends Error {
	readonly code: ErrorCode;

	constructor(response: ErrorResponse) {
		super(response.message);
		this.name = "RbtrDaemonError";
		this.code = response.code;
	}
}

/**
 * Returned when ``rbtr daemon status --json`` succeeds.
 *
 * The shape is the generated ``DaemonStatusReport``.
 */
export type DaemonStatus = DaemonStatusReport;

export interface SendOptions {
	/** Milliseconds to wait for the daemon's reply.  Default 10 000. */
	receiveTimeout?: number;
	/** Milliseconds to wait for the outbound send.  Default 5 000. */
	sendTimeout?: number;
}

/** Narrow ``Response`` to the variant produced by a request of *kind* K. */
type ResponseFor<K extends string> = Extract<Response, { kind: K }>;

/**
 * Ask the CLI for the running daemon's status.
 *
 * The path to the status file depends on the user's `RBTR_HOME`
 * setting, so the CLI is the source of truth — we do not read
 * the JSON file directly.
 */
export async function queryDaemonStatus(): Promise<DaemonStatus> {
	return new Promise<DaemonStatus>((resolve, reject) => {
		const proc = spawn("rbtr", ["--json", "daemon", "status"], {
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
 * Send one request to the daemon and return the matching response.
 *
 * Throws ``RbtrDaemonError`` when the daemon replies with an
 * ``ErrorResponse`` — on success the returned type is narrowed
 * to the response variant matching the request's ``kind``.
 *
 * A caller that does not want to shell out to the CLI on every
 * call can pass an explicit *rpcEndpoint* (an ``ipc://`` URI).
 */
export async function send<R extends Request>(
	request: R,
	options: SendOptions & { rpcEndpoint?: string } = {},
): Promise<ResponseFor<R["kind"]>> {
	const endpoint = options.rpcEndpoint ?? (await queryDaemonStatus()).rpc;
	if (!endpoint) {
		throw new Error("daemon has no rpc endpoint (not running?)");
	}

	const sock = new ZmqRequest({
		receiveTimeout: options.receiveTimeout ?? 10_000,
		sendTimeout: options.sendTimeout ?? 5_000,
	});

	try {
		sock.connect(endpoint);
		await sock.send(JSON.stringify(request));
		const [reply] = await sock.receive();
		const response = JSON.parse(reply.toString()) as Response;

		if (response.kind === "error") {
			throw new RbtrDaemonError(response);
		}

		// Discriminator guarantees the kind matches the request's
		// kind for every non-error response (the server is typed).
		return response as ResponseFor<R["kind"]>;
	} finally {
		sock.close();
	}
}
