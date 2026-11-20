#!/usr/bin/env -S node --no-warnings
/**
 * Stands in for the program it is linked as: appends its command line to
 * $RECORDED_PROCESSES, when set, then runs that program from the rest of
 * PATH, and exits as it does. A command line $SLOW_COMMANDS (a JSON object)
 * maps to milliseconds waits that long first, as a slow start would.
 */

import { spawn } from "node:child_process";
import { appendFileSync } from "node:fs";
import { basename, delimiter, dirname } from "node:path";
import { setTimeout } from "node:timers/promises";

const name = basename(process.argv[1]);
const args = process.argv.slice(2);
const log = process.env.RECORDED_PROCESSES;
const line = [name, ...args].join(" ");
if (log) appendFileSync(log, `${line}\n`);
const slow = JSON.parse(process.env.SLOW_COMMANDS ?? "{}") as Record<string, number>;
if (line in slow) await setTimeout(slow[line]);

const here = dirname(process.argv[1]);
const path = (process.env.PATH ?? "")
  .split(delimiter)
  .filter((dir) => dir !== here)
  .join(delimiter);
const child = spawn(name, args, { stdio: "inherit", env: { ...process.env, PATH: path } });
for (const signal of ["SIGTERM", "SIGINT"] as const) process.on(signal, () => child.kill(signal));
child.on("error", () => process.exit(127));
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
