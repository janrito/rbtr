/**
 * Scenarios for ``daemon-client.test.ts``.
 *
 * Data-first: each case names a behaviour and spells out its
 * inputs and expected outputs.  The test file is thin and
 * iterates over these cases with ``test.each``.
 */

import type { ErrorCode, Request, Response } from "../extensions/rbtr/generated/protocol.js";

export interface SendScenario {
	/** Human-readable name (used as the test case ID). */
	readonly name: string;
	/** The request the client sends. */
	readonly request: Request;
	/** What the fake daemon replies with. */
	readonly daemonResponse: Response;
	/** What we expect to observe back at the caller. */
	readonly expected:
		| { kind: "success"; response: Response }
		| { kind: "throws"; code: ErrorCode; messageContains: string };
}

export const sendScenarios: readonly SendScenario[] = [
	{
		name: "status round-trip",
		request: { kind: "status", repo: "/r" },
		daemonResponse: {
			kind: "status",
			exists: true,
			db_path: "/home/db.duckdb",
			total_chunks: 123,
			indexed_refs: ["abc123"],
		},
		expected: {
			kind: "success",
			response: {
				kind: "status",
				exists: true,
				db_path: "/home/db.duckdb",
				total_chunks: 123,
				indexed_refs: ["abc123"],
			},
		},
	},
	{
		name: "search round-trip",
		request: { kind: "search", repo: "/r", query: "retry logic" },
		daemonResponse: { kind: "search", results: [] },
		expected: { kind: "success", response: { kind: "search", results: [] } },
	},
	{
		name: "index submission returns ok",
		request: { kind: "index", repo: "/r", refs: ["HEAD"] },
		daemonResponse: { kind: "ok" },
		expected: { kind: "success", response: { kind: "ok" } },
	},
	{
		name: "error response raises RbtrDaemonError with code",
		request: { kind: "status", repo: "/nope" },
		daemonResponse: {
			kind: "error",
			code: "repo_not_found",
			message: "/nope is not a git repository",
		},
		expected: {
			kind: "throws",
			code: "repo_not_found",
			messageContains: "not a git repository",
		},
	},
	{
		name: "error response preserves INTERNAL code",
		request: { kind: "search", repo: "/r", query: "x" },
		daemonResponse: {
			kind: "error",
			code: "internal",
			message: "handler crashed",
		},
		expected: {
			kind: "throws",
			code: "internal",
			messageContains: "handler crashed",
		},
	},
];
