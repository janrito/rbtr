#!/usr/bin/env -S node --no-warnings
/**
 * Stands in for `rbtr` as a `command` setting that knows only one daemon:
 * appends its command line to $RECORDED_PROCESSES, when set; `daemon
 * status` prints $RBTR_DAEMON_STATUS, a `DaemonStatusReport` (a stopped
 * daemon when unset); any other command fails, as an rbtr that cannot reach
 * its index would.
 */

import { appendFileSync } from "node:fs";

const log = process.env.RECORDED_PROCESSES;
if (log) appendFileSync(log, `${["daemon-at", ...process.argv.slice(2)].join(" ")}\n`);

const args = process.argv.slice(2).filter((arg) => arg !== "--json");
if (args.join(" ") === "daemon status") {
  process.stdout.write(`${process.env.RBTR_DAEMON_STATUS ?? JSON.stringify({ running: false })}\n`);
  process.exit(0);
}
process.stderr.write(`daemon-at: ${args.join(" ")}: no index here\n`);
process.exit(1);
