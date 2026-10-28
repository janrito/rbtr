/**
 * Build the run's repository and its index once per run, choose the code
 * the tests use from rbtr's own replies, and start the run's shared daemon
 * over a copy of that index. Nothing is embedded (`rbtr-env.ts`).
 *
 * The repository is a clone of this checkout, so the tests run over real
 * code and name none of it: `corpus` holds what they need, chosen by the
 * rules below from the checkout's Python files, as this checkout is mostly
 * Python. Its history: `base` is the checkout's HEAD; `main` (HEAD) changes
 * the subject function by one line and deletes any `.pi/`; `unindexed` adds
 * a commit on top of `main`. `HEAD` and `base` are indexed, `unindexed` never is.
 *
 * Tests read the repository and never write to it. A test that writes
 * project settings, changes the watch set or watches a build works on a
 * repository of its own (`writableRepo` in `session.fixtures.ts`). A test
 * that stops, starts or freezes a daemon gets its own (`ownDaemon`), over
 * another copy of the pristine index.
 *
 * Everything a run creates lives under one root, which records the run's
 * pid; the data directories of the daemons it starts are listed in the
 * root's `daemons` file. Teardown, Ctrl-C and SIGTERM stop those daemons,
 * and the next run stops the daemons of a root whose run is gone, so an
 * interrupted run leaves none behind, frozen ones included.
 */

import { execFileSync } from "node:child_process";
import {
  appendFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { VERSION } from "@earendil-works/pi-coding-agent";
import type { TestProject } from "vitest/node";

import { OUTLINE_FROM_LINES, searchedName } from "../extensions/rbtr/annotate.js";
import { send } from "../extensions/rbtr/daemon-client.js";
import { TEST_RBTR_ENV } from "./rbtr-env.js";

/** What the tests use from the run's repository, chosen from rbtr's replies. */
export interface Corpus {
  /** The run's repository: a clone of this checkout. */
  repo: string;
  /**
   * A function on a plain `def` line, in a file of fewer than 200 lines,
   * whose name a search note looks up, and which another file uses: its
   * name, the file defining it, and a file using it. `main` changes it by
   * one line.
   */
  subject: { name: string; file: string; user: string };
  /** A file of 200 lines or more, which a first read outlines. */
  long: string;
}

declare module "vitest" {
  export interface ProvidedContext {
    /** The version of the installed pi, which the tests run against (`vitest.config.ts`). */
    piVersion: string;
    /** The run's repository, what the tests use from it, and a data directory holding its index, which no daemon opens. */
    corpus: Corpus & { pristine: string };
    /** The shared daemon: its `RBTR_*_DIR` settings and its RPC endpoint. */
    shared: { dirs: Record<string, string>; rpc: string };
    /** The run's root directory, and the file listing its daemons' data directories. */
    run: { root: string; daemons: string };
  }
}

const RUN_PREFIX = "rbtr-test-";

/** Whether a process with *pid* is alive. */
function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** How long cleanup waits for one `rbtr` call, so it cannot hang a run. */
const CLEANUP_CALL_MS = 30_000;

/**
 * Stop the daemons listed under *root*: a frozen one is continued first,
 * and one that `daemon stop` does not end in time is killed.
 */
function stopDaemons(root: string): void {
  const listed = join(root, "daemons");
  if (!existsSync(listed)) return;
  const dataDirs = readFileSync(listed, "utf8")
    .split("\n")
    .filter((line) => line !== "");
  for (const dataDir of dataDirs) {
    const env = { ...process.env, ...TEST_RBTR_ENV, RBTR_DATA_DIR: dataDir };
    const status = JSON.parse(
      execFileSync("rbtr", ["--json", "daemon", "status"], { env, encoding: "utf8", timeout: CLEANUP_CALL_MS }),
    ) as { running: boolean; pid?: number | null };
    if (!status.running || status.pid == null) continue;
    process.kill(status.pid, "SIGCONT");
    try {
      execFileSync("rbtr", ["--json", "daemon", "stop"], { env, stdio: "ignore", timeout: CLEANUP_CALL_MS });
    } catch {
      if (isAlive(status.pid)) process.kill(status.pid, "SIGKILL");
    }
  }
}

/** Stop the daemons of every earlier run whose process is gone, and remove its root. */
function sweepAbandonedRuns(): void {
  for (const name of readdirSync(tmpdir()).filter((entry) => entry.startsWith(RUN_PREFIX))) {
    const root = join(tmpdir(), name);
    const pidFile = join(root, "run.pid");
    if (existsSync(pidFile) && isAlive(Number(readFileSync(pidFile, "utf8")))) continue;
    stopDaemons(root);
    rmSync(root, { recursive: true, force: true });
  }
}

/** Resolves once *ready* holds, checking every 100 ms; rejects after 60 s. */
async function until(ready: () => Promise<boolean>): Promise<void> {
  for (const deadline = Date.now() + 60_000; !(await ready()); ) {
    if (Date.now() > deadline) throw new Error("the run's index was not ready within 60 s");
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

/**
 * Choose the corpus from *repo*'s Python files and the replies of the
 * daemon at *rpc*, which has *repo*'s HEAD indexed. Files are tried in path
 * order, so a checkout gives the same choice every run.
 */
async function chooseCorpus(repo: string, rpc: string): Promise<Corpus> {
  const python = execFileSync("git", ["ls-files", "*.py"], { cwd: repo, encoding: "utf8" })
    .split("\n")
    .filter((file) => file !== "")
    .sort();
  const lines = (file: string) => readFileSync(join(repo, file), "utf8").split("\n");
  const long = python.find((file) => lines(file).length >= OUTLINE_FROM_LINES);
  if (long === undefined) throw new Error("the checkout holds no Python file of 200 lines or more for the tests");
  for (const file of python.filter((file) => lines(file).length < OUTLINE_FROM_LINES)) {
    const listed = await send({ kind: "list_symbols", repo_path: repo, file_path: file }, { rpcEndpoint: rpc });
    for (const chunk of listed.chunks) {
      if (chunk.kind !== "function" || searchedName(chunk.name) !== chunk.name) continue;
      if (!lines(file)[chunk.line_start - 1].trimStart().startsWith(`def ${chunk.name}(`)) continue;
      const refs = await send(
        { kind: "find_refs", repo_path: repo, symbol: chunk.name, file_paths: [file] },
        { rpcEndpoint: rpc },
      );
      const user = refs.refs.map((ref) => ref.file_path).find((path) => path !== file);
      if (user !== undefined) return { repo, subject: { name: chunk.name, file, user }, long };
    }
  }
  throw new Error("the checkout holds no function the tests can take as their subject");
}

export default async function setup(project: TestProject): Promise<() => void> {
  const { piVersion } = project.getProvidedContext();
  if (VERSION !== piVersion) throw new Error(`the tests loaded pi ${VERSION}, not the installed pi ${piVersion}`);
  sweepAbandonedRuns();
  const root = mkdtempSync(join(tmpdir(), RUN_PREFIX));
  writeFileSync(join(root, "run.pid"), String(process.pid));
  const daemons = join(root, "daemons");
  const here = dirname(fileURLToPath(import.meta.url));
  const checkout = execFileSync("git", ["rev-parse", "--show-toplevel"], { cwd: here, encoding: "utf8" }).trim();
  const repo = join(root, "repo");
  // Git LFS files stay pointers: a checkout made without LFS objects, as
  // CI's is, cannot fill them in a clone, and no test reads them.
  execFileSync("git", ["clone", "-q", checkout, repo], { env: { ...process.env, GIT_LFS_SKIP_SMUDGE: "1" } });
  const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, encoding: "utf8" });
  git("switch", "-qC", "main");
  git("config", "user.email", "test@example.com");
  git("config", "user.name", "test");
  git("branch", "base");

  const pristine = {
    RBTR_DATA_DIR: join(root, "pristine"),
    RBTR_CONFIG_DIR: join(root, "config"),
    RBTR_LOG_DIR: join(root, "logs"),
  };
  for (const dir of Object.values(pristine)) mkdirSync(dir);
  const rbtr = (dirs: Record<string, string>, ...args: string[]) =>
    execFileSync("rbtr", ["--json", ...args], { env: { ...process.env, ...TEST_RBTR_ENV, ...dirs }, encoding: "utf8" });
  rbtr(pristine, "watch", "--no-daemon", "--repo-path", repo, "HEAD");

  // A daemon over the pristine index answers the questions the corpus is
  // chosen by, then indexes the history built from it, and stops.
  appendFileSync(daemons, `${pristine.RBTR_DATA_DIR}\n`);
  const chooser = JSON.parse(rbtr(pristine, "daemon", "start")) as { rpc: string };
  const corpus = await chooseCorpus(repo, chooser.rpc);
  const file = join(repo, corpus.subject.file);
  const text = readFileSync(file, "utf8").split("\n");
  const at = text.findIndex((line) => line.trimStart().startsWith(`def ${corpus.subject.name}(`));
  const indent = /^\s*/.exec(text[at + 1])?.[0] ?? "";
  text.splice(at + 1, 0, `${indent}# The line the tests' main commit adds.`);
  writeFileSync(file, text.join("\n"));
  // pi and the extension read a project's `.pi/`; the tests write their own.
  git("rm", "-rq", "--ignore-unmatch", ".pi");
  git("commit", "-qam", "Change the subject function by one line");
  git("switch", "-qc", "unindexed");
  const copy = join(dirname(corpus.subject.file), `unindexed_${basename(corpus.subject.file)}`);
  cpSync(file, join(repo, copy));
  git("add", ".");
  git("commit", "-qm", "Add a copy of the subject's file");
  git("switch", "-q", "main");
  await send({ kind: "watch", repo_path: repo, refs: ["HEAD", "base"] }, { rpcEndpoint: chooser.rpc });
  await until(async () => {
    const status = await send({ kind: "status", repo_path: repo }, { rpcEndpoint: chooser.rpc });
    const names = (status.indexed_refs ?? []).flatMap((ref) => ref.names ?? []);
    return names.includes("HEAD") && names.includes("base");
  });
  rbtr(pristine, "daemon", "stop");

  const shared = { ...pristine, RBTR_DATA_DIR: join(root, "shared") };
  cpSync(pristine.RBTR_DATA_DIR, shared.RBTR_DATA_DIR, { recursive: true });
  appendFileSync(daemons, `${shared.RBTR_DATA_DIR}\n`);
  const { rpc } = JSON.parse(rbtr(shared, "daemon", "start")) as { rpc: string };
  project.provide("corpus", { ...corpus, pristine: pristine.RBTR_DATA_DIR });
  project.provide("shared", { dirs: shared, rpc });
  project.provide("run", { root, daemons });

  // On a signal, before vitest's own handlers end the process: workers may
  // still be writing under the root, so the next run's sweep removes it.
  const onSignal = () => stopDaemons(root);
  for (const signal of ["SIGINT", "SIGTERM"] as const) process.prependListener(signal, onSignal);
  return () => {
    for (const signal of ["SIGINT", "SIGTERM"] as const) process.off(signal, onSignal);
    stopDaemons(root);
    rmSync(root, { recursive: true, force: true });
  };
}
