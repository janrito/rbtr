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
import { Publisher as ZmqPublisher, Reply as ZmqReply } from "zeromq";

import type { Notification, Request, Response } from "../../extensions/rbtr/generated/protocol.js";

/**
 * A canned reply, or a function that computes the reply from the
 * request.  The function form exists for tests that want to assert
 * on the request's shape.
 */
export type FakeReply = Response | ((request: Request) => Response);

export interface FakeDaemonHandle extends AsyncDisposable {
	/** ``ipc://...`` REP endpoint (for ``send()``). */
	readonly endpoint: string;
	/** ``ipc://...`` PUB endpoint (for ``DaemonSession.subscribe()``). */
	readonly pubEndpoint: string;
	/** Every request the fake has received, in order. */
	readonly received: readonly Request[];
	/**
	 * Publish a notification on the PUB socket.
	 *
	 * A short delay after creation is usually needed before the
	 * first subscriber receives anything (PUB/SUB slow joiner);
	 * tests that rely on timing should sleep ~50 ms after
	 * subscribing.
	 */
	publish(notification: Notification): Promise<void>;
}

/**
 * Start a FakeDaemon.  It serves *reply* for every request it
 * receives and records each request in ``received``.
 *
 * The socket is cleaned up when the returned handle is disposed
 * (``await using`` / explicit ``[Symbol.asyncDispose]``).
 */
export async function startFakeDaemon(opts: { reply?: FakeReply } = {}): Promise<FakeDaemonHandle> {
	// macOS caps unix-domain socket paths at ~104 bytes.
	// Keep the endpoint short: 8 random hex chars, no suffix.
	const dir = mkdtempSync(join(tmpdir(), "rbtr-fd-"));
	const token = randomBytes(4).toString("hex");
	const endpoint = `ipc://${dir}/${token}.rpc`;
	const pubEndpoint = `ipc://${dir}/${token}.pub`;

	const rep = new ZmqReply();
	await rep.bind(endpoint);
	const pub = new ZmqPublisher();
	await pub.bind(pubEndpoint);

	const received: Request[] = [];
	let stopped = false;

	// Serve requests in the background until disposed.
	void (async () => {
		while (!stopped) {
			let rawRequest: Buffer;
			try {
				[rawRequest] = await rep.receive();
			} catch {
				return; // socket closed
			}
			const request = JSON.parse(rawRequest.toString()) as Request;
			received.push(request);
			if (opts.reply === undefined) continue;
			const response = typeof opts.reply === "function" ? opts.reply(request) : opts.reply;
			await rep.send(JSON.stringify(response));
		}
	})();

	return {
		endpoint,
		pubEndpoint,
		received,
		async publish(notification: Notification) {
			await pub.send(JSON.stringify(notification));
		},
		async [Symbol.asyncDispose]() {
			stopped = true;
			rep.close();
			pub.close();
		},
	};
}
