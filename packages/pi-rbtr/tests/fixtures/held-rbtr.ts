#!/usr/bin/env -S node --no-warnings
/**
 * Stands in for `rbtr` as a `command` setting: appends its command line to
 * $RECORDED_PROCESSES, when set, then runs $RBTR_HELD_PROGRAM (`rbtr` when
 * unset), and exits as it does. A command line holding the word
 * $RBTR_HOLD_ON (every command line, when it is empty) waits first, until
 * the file $RBTR_RELEASE names exists.
 */

import { spawn } from "node:child_process";
import { appendFileSync, existsSync } from "node:fs";
import { setTimeout } from "node:timers/promises";

const args = process.argv.slice(2);
const log = process.env.RECORDED_PROCESSES;
if (log) appendFileSync(log, `${["held-rbtr", ...args].join(" ")}\n`);

const holdOn = process.env.RBTR_HOLD_ON ?? "";
const release = process.env.RBTR_RELEASE ?? "";
if (holdOn === "" || args.includes(holdOn)) {
  while (!existsSync(release)) await setTimeout(50);
}

const child = spawn(process.env.RBTR_HELD_PROGRAM || "rbtr", args, { stdio: "inherit" });
for (const signal of ["SIGTERM", "SIGINT"] as const) process.on(signal, () => child.kill(signal));
child.on("error", () => process.exit(127));
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
