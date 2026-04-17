/**
 * Scenarios for ``daemon-session.test.ts``.
 */

import type { Request, Response } from "../extensions/rbtr/generated/protocol.js";

export interface SessionScenario {
	readonly name: string;
	readonly request: Request;
	readonly daemonResponse: Response;
	readonly expectedResponse: Response;
}

export const sessionScenarios: readonly SessionScenario[] = [
	{
		name: "session routes status through cached endpoint",
		request: { kind: "status", repo: "/r" },
		daemonResponse: {
			kind: "status",
			exists: true,
			db_path: "/home/db.duckdb",
			total_chunks: 1,
			indexed_refs: [],
		},
		expectedResponse: {
			kind: "status",
			exists: true,
			db_path: "/home/db.duckdb",
			total_chunks: 1,
			indexed_refs: [],
		},
	},
	{
		name: "session routes search through cached endpoint",
		request: { kind: "search", repo: "/r", query: "hello" },
		daemonResponse: { kind: "search", results: [] },
		expectedResponse: { kind: "search", results: [] },
	},
	{
		name: "session fires index as fire-and-forget ok",
		request: { kind: "index", repo: "/r", refs: ["HEAD"] },
		daemonResponse: { kind: "ok" },
		expectedResponse: { kind: "ok" },
	},
];
