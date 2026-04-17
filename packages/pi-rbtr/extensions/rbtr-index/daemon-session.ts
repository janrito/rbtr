/**
 * Session-level state for the daemon client.
 *
 * The extension reuses one `DaemonSession` across tool calls
 * within a pi session.  It caches the RPC endpoint (set at
 * `session_start`) and the PUB endpoint (so Phase 8.4 can
 * attach a subscriber), invalidating on transport errors.
 */

import { Subscriber as ZmqSubscriber } from "zeromq";

import { type DaemonStatus, queryDaemonStatus, send } from "./daemon-client.js";
import type { Notification, Request, Response } from "./generated/protocol.js";

/**
 * A transport-level failure talking to the daemon — socket
 * refused, send/recv timeout, JSON parse.  Distinct from
 * ``RbtrDaemonError`` which means the daemon replied with a
 * typed error code.
 */
export class DaemonUnavailableError extends Error {
	constructor(cause: unknown) {
		const detail = cause instanceof Error ? cause.message : String(cause);
		super(`daemon unavailable: ${detail}`);
		this.name = "DaemonUnavailableError";
		this.cause = cause;
	}
}

/** Narrow ``Response`` to the variant for a request of *kind* K. */
type ResponseFor<K extends string> = Extract<Response, { kind: K }>;

/** Called once per notification received on the SUB socket. */
export type NotificationHandler = (notification: Notification) => void;

export class DaemonSession {
	private status: DaemonStatus | null = null;
	private subscriber: ZmqSubscriber | null = null;

	/** ``true`` if the daemon has been reachable at least once this session. */
	get available(): boolean {
		return this.status !== null && this.status.running === true;
	}

	/** Last known RPC endpoint (``ipc://...``), or ``null`` if unavailable. */
	get rpcEndpoint(): string | null {
		return this.status?.rpc ?? null;
	}

	/** Last known PUB endpoint (``ipc://...``), or ``null`` if unavailable. */
	get pubEndpoint(): string | null {
		return this.status?.pub ?? null;
	}

	/**
	 * Refresh status from the CLI.  Caches the result on success.
	 *
	 * Any failure (CLI not on PATH, daemon not running) leaves the
	 * session marked unavailable so tools fall back to CLI exec.
	 */
	async refresh(): Promise<DaemonStatus | null> {
		try {
			this.status = await queryDaemonStatus();
		} catch {
			this.status = null;
		}
		return this.status;
	}

	/**
	 * Send a request through the cached RPC endpoint.
	 *
	 * Re-queries once on transport failure (covers daemon restart
	 * with a new endpoint) before giving up with
	 * ``DaemonUnavailableError``.  ``RbtrDaemonError`` from the
	 * daemon itself is propagated as-is — the caller decides what
	 * to do with it.
	 */
	async send<R extends Request>(request: R): Promise<ResponseFor<R["kind"]>> {
		if (!this.rpcEndpoint) {
			await this.refresh();
			if (!this.rpcEndpoint) {
				throw new DaemonUnavailableError("no rpc endpoint");
			}
		}

		try {
			return (await send(request, { rpcEndpoint: this.rpcEndpoint })) as ResponseFor<R["kind"]>;
		} catch (err) {
			if (err instanceof Error && err.name === "RbtrDaemonError") {
				throw err;
			}
			// Transport failure — try one refresh + retry before
			// giving up.  Covers daemon restart with a new endpoint.
			this.status = null;
			await this.refresh();
			if (!this.rpcEndpoint) {
				throw new DaemonUnavailableError(err);
			}
			try {
				return (await send(request, { rpcEndpoint: this.rpcEndpoint })) as ResponseFor<R["kind"]>;
			} catch (err2) {
				if (err2 instanceof Error && err2.name === "RbtrDaemonError") {
					throw err2;
				}
				throw new DaemonUnavailableError(err2);
			}
		}
	}

	/**
	 * Subscribe to the daemon's PUB stream for the session.
	 *
	 * *handler* is called once per deserialised ``Notification``.
	 * Callers typically filter on ``notification.repo === cwd``
	 * themselves — the daemon publishes for all watched repos.
	 *
	 * Returns a disposer.  Calling it (or ``stopSubscribing()``)
	 * closes the SUB socket.  Safe to call twice.
	 */
	subscribe(handler: NotificationHandler): () => void {
		if (this.subscriber !== null) {
			return () => {
				this.stopSubscribing();
			};
		}
		if (!this.pubEndpoint) {
			throw new DaemonUnavailableError("no pub endpoint");
		}

		const sub = new ZmqSubscriber();
		sub.connect(this.pubEndpoint);
		sub.subscribe(); // empty prefix — server publishes untopiced frames
		this.subscriber = sub;

		void (async () => {
			for await (const [frame] of sub) {
				let parsed: Notification;
				try {
					parsed = JSON.parse(frame.toString()) as Notification;
				} catch {
					continue; // malformed frame — skip
				}
				try {
					handler(parsed);
				} catch {
					// Handler errors must not kill the loop.
				}
			}
		})();

		return () => {
			this.stopSubscribing();
		};
	}

	stopSubscribing(): void {
		if (this.subscriber === null) return;
		this.subscriber.close();
		this.subscriber = null;
	}
}
