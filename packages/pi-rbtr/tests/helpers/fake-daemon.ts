/**
 * FakeDaemon — a minimal ZMQ REP counterpart for ``DaemonClient``.
 *
 * Not a mock of the client: a real ZMQ reply socket bound to a
 * disposable IPC endpoint.  Tests drive the production
 * ``send()`` function and this counterpart acts as the daemon.
 *
 * Uses ``await using`` for automatic socket cleanup via the
 * explicit-resource-management proposal (ECMAScript 2026, available
 * in Node 22+ / bun).  Callers write::
 *
 *     await using daemon = await startFakeDaemon({ reply: ... });
 *     // socket bound at daemon.endpoint
 *
 * and the socket is closed when the block exits, regardless of
 * thrown errors.
 */

import { randomBytes } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Reply as ZmqReply } from "zeromq";

import type { Request, Response } from "../../extensions/rbtr-index/generated/protocol.js";

/**
 * A canned reply, or a function that computes the reply from the
 * request.  The function form exists for tests that want to assert
 * on the request's shape.
 */
export type FakeReply = Response | ((request: Request) => Response);

export interface FakeDaemonHandle extends AsyncDisposable {
	/** ``ipc://...`` endpoint to pass as ``rpcEndpoint`` to ``send()``. */
	readonly endpoint: string;
	/** Every request the fake has received, in order. */
	readonly received: readonly Request[];
}

/**
 * Start a FakeDaemon.  It serves *reply* for every request it
 * receives and records each request in ``received``.
 *
 * The socket is cleaned up when the returned handle is disposed
 * (``await using`` / explicit ``[Symbol.asyncDispose]``).
 */
export async function startFakeDaemon(opts: { reply: FakeReply }): Promise<FakeDaemonHandle> {
	// macOS caps unix-domain socket paths at ~104 bytes.
	// Keep the endpoint short: 8 random hex chars, no suffix.
	const dir = mkdtempSync(join(tmpdir(), "rbtr-fd-"));
	const endpoint = `ipc://${dir}/${randomBytes(4).toString("hex")}.rpc`;
	const sock = new ZmqReply();
	await sock.bind(endpoint);

	const received: Request[] = [];
	let stopped = false;

	// Serve requests in the background until disposed.
	void (async () => {
		while (!stopped) {
			let rawRequest: Buffer;
			try {
				[rawRequest] = await sock.receive();
			} catch {
				return; // socket closed
			}
			const request = JSON.parse(rawRequest.toString()) as Request;
			received.push(request);
			const response = typeof opts.reply === "function" ? opts.reply(request) : opts.reply;
			await sock.send(JSON.stringify(response));
		}
	})();

	return {
		endpoint,
		received,
		async [Symbol.asyncDispose]() {
			stopped = true;
			sock.close();
		},
	};
}
