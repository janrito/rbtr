/**
 * Tests for DaemonSession — caching, retry, unavailable semantics.
 *
 * Uses FakeDaemon (real ZMQ on both sides) and drives the
 * session via its public ``send()``.  The rpc endpoint is
 * pre-seeded so there is no CLI shell-out during the test.
 */

import { describe, expect, test } from "vitest";

import { RbtrDaemonError } from "../extensions/rbtr-index/daemon-client.js";
import { DaemonSession, DaemonUnavailableError } from "../extensions/rbtr-index/daemon-session.js";
import { sessionScenarios } from "./daemon-session.cases.js";
import { startFakeDaemon } from "./helpers/fake-daemon.js";

/**
 * Build a DaemonSession already primed with the endpoint of a
 * running fake daemon.  Tests never need the CLI path.
 */
function primedSession(endpoint: string): DaemonSession {
	const session = new DaemonSession();
	// @ts-expect-error  —  write-through to private cache for
	// tests; prevents a CLI round-trip on first send().
	session.status = { running: true, rpc: endpoint, pub: null };
	return session;
}

describe("DaemonSession.send", () => {
	test.each(sessionScenarios)("$name", async ({ request, daemonResponse, expectedResponse }) => {
		await using daemon = await startFakeDaemon({ reply: daemonResponse });
		const session = primedSession(daemon.endpoint);

		const response = await session.send(request);
		expect(response).toEqual(expectedResponse);
		expect(daemon.received).toEqual([request]);
	});

	test("ErrorResponse from the daemon surfaces as RbtrDaemonError", async () => {
		await using daemon = await startFakeDaemon({
			reply: {
				kind: "error",
				code: "repo_not_found",
				message: "/nope is not a git repository",
			},
		});
		const session = primedSession(daemon.endpoint);

		await expect(session.send({ kind: "status", repo: "/nope" })).rejects.toSatisfy(
			(err: unknown) =>
				err instanceof RbtrDaemonError && err.code === "repo_not_found" && err.message.includes("not a git repository"),
		);
	});

	test("sends without cached endpoint raise DaemonUnavailableError before any CLI call", async () => {
		const session = new DaemonSession();
		// Not primed.  refresh() will shell out to rbtr daemon status,
		// which returns a not-running report in our isolated PATH.
		await expect(session.send({ kind: "shutdown" })).rejects.toBeInstanceOf(DaemonUnavailableError);
	});
});
