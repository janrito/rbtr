#!/usr/bin/env -S node --no-warnings
/**
 * Stands in for a program that is not installed: exits as a shell does when
 * it cannot find a command.
 */

import { basename } from "node:path";

process.stderr.write(`${basename(process.argv[1])}: command not found\n`);
process.exit(127);
