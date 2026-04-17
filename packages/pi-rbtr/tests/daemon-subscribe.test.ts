/**
 * Tests for DaemonSession.subscribe — real PUB socket on the
 * fake daemon side, real SUB socket inside the session.  No
 * mocking of ZMQ.
 */

import { describe, expect, test } from "vitest";

import { DaemonSession } from "../extensions/rbtr/daemon-session.js";
import type { Notification } from "../extensions/rbtr/generated/protocol.js";
import { startFakeDaemon } from "./helpers/fake-daemon.js";

/**
 * Prime a session with both endpoints of a fake daemon so no
 * CLI round-trip is needed.
 */
function primedSession(rpc: string, pub: string): DaemonSession {
	const session = new DaemonSession();
	// @ts-expect-error  —  private write-through
	session.status = { running: true, rpc, pub };
	return session;
}

// Publishers + subscribers in ZMQ have a 'slow joiner' problem —
// the subscriber needs a brief moment to attach before the
// publisher's messages flow.  Sleep a beat between subscribe
// and publish.
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

describe("DaemonSession.subscribe", () => {
	test("forwards notifications to the handler", async () => {
		await using daemon = await startFakeDaemon();
		const session = primedSession(daemon.endpoint, daemon.pubEndpoint);

		const received: Notification[] = [];
		const dispose = session.subscribe((n) => {
			received.push(n);
		});

		await sleep(80);
		await daemon.publish({
			kind: "progress",
			repo: "/r",
			phase: "parsing",
			current: 10,
			total: 42,
		});
		await sleep(80);
		await daemon.publish({
			kind: "ready",
			repo: "/r",
			ref: "abc",
			chunks: 100,
			edges: 200,
			elapsed: 1.5,
		});
		await sleep(80);

		dispose();
		expect(received.map((n) => n.kind)).toEqual(["progress", "ready"]);
	});

	test("stopSubscribing is idempotent", async () => {
		await using daemon = await startFakeDaemon();
		const session = primedSession(daemon.endpoint, daemon.pubEndpoint);

		session.subscribe(() => {});
		session.stopSubscribing();
		// Second call is a no-op, not a throw.
		expect(() => {
			session.stopSubscribing();
		}).not.toThrow();
	});

	test("calling subscribe twice returns a disposer for the existing subscription", async () => {
		await using daemon = await startFakeDaemon();
		const session = primedSession(daemon.endpoint, daemon.pubEndpoint);

		const received: Notification[] = [];
		const dispose1 = session.subscribe((n) => {
			received.push(n);
		});
		// Second call should be a no-op that returns a disposer
		// pointing at the same active subscription.
		const dispose2 = session.subscribe(() => {
			throw new Error("second handler must not be attached");
		});

		await sleep(80);
		await daemon.publish({
			kind: "index_error",
			repo: "/r",
			message: "boom",
		});
		await sleep(80);

		expect(received).toHaveLength(1);
		dispose1();
		dispose2();
	});
});
