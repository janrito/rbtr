/**
 * Tests for the DaemonClient.  Drives the real ``send`` function
 * against a real ZMQ reply socket (``FakeDaemon``) — no internal
 * mocking.
 */

import { describe, expect, test } from "vitest";

import { RbtrDaemonError, send } from "../extensions/rbtr-index/daemon-client.js";
import { sendScenarios } from "./daemon-client.cases.js";
import { startFakeDaemon } from "./helpers/fake-daemon.js";

describe("send", () => {
	test.each(sendScenarios)("$name", async ({ request, daemonResponse, expected }) => {
		await using daemon = await startFakeDaemon({ reply: daemonResponse });

		if (expected.kind === "throws") {
			await expect(send(request, { rpcEndpoint: daemon.endpoint })).rejects.toSatisfy(
				(err: unknown) =>
					err instanceof RbtrDaemonError &&
					err.code === expected.code &&
					err.message.includes(expected.messageContains),
			);
		} else {
			const response = await send(request, { rpcEndpoint: daemon.endpoint });
			expect(response).toEqual(expected.response);
		}

		// Every scenario sends exactly one request; the fake must
		// have seen it.
		expect(daemon.received).toEqual([request]);
	});

	test("multiple sequential requests reuse no socket state", async () => {
		await using daemon = await startFakeDaemon({ reply: { kind: "ok" } });

		await send({ kind: "shutdown" }, { rpcEndpoint: daemon.endpoint });
		await send({ kind: "shutdown" }, { rpcEndpoint: daemon.endpoint });
		await send({ kind: "shutdown" }, { rpcEndpoint: daemon.endpoint });

		expect(daemon.received).toHaveLength(3);
	});

	test("reply function can inspect the request", async () => {
		let seen: unknown = null;
		await using daemon = await startFakeDaemon({
			reply: (req) => {
				seen = req;
				return { kind: "ok" };
			},
		});

		await send({ kind: "search", repo: "/r", query: "needle", limit: 5 }, { rpcEndpoint: daemon.endpoint });

		expect(seen).toEqual({ kind: "search", repo: "/r", query: "needle", limit: 5 });
	});
});
