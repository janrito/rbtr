/**
 * Fixtures that run pi-rbtr the way pi runs it.
 *
 * A test gets pi's own session runtime (in-memory session and settings),
 * pi-ai's faux model to script the assistant's tool calls, and a real
 * `rbtr` daemon, isolated from the user's, over a real git repository: a
 * clone of this checkout. The stand-ins are at pi's UI boundary: an
 * `ExtensionUIContext` that records notifications and footer statuses and
 * answers dialogs, and the footer's data (the statuses, no git branch),
 * from which pi's real `FooterComponent` renders the footer.
 *
 * The daemon is the run's shared one, started by `global-setup.ts`, unless a
 * test asks for its own (`ownDaemon`). File-scoped fixtures (`shared`,
 * `corpus`, `repo`, `theme`) are built once per test file; the rest per
 * test. Each file runs in its own process, so the environment a file sets
 * is its own. The repository, its history and the code the tests use from
 * it (`corpus`) are described in `global-setup.ts`, which builds them once
 * per run.
 */

import { execFileSync } from "node:child_process";
import {
  appendFileSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { basename, delimiter, dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  type FauxProviderHandle,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxToolCall,
  type JsonObject,
} from "@earendil-works/pi-ai";
import {
  type AgentSessionRuntime,
  type AgentToolResult,
  createAgentSessionFromServices,
  createAgentSessionRuntime,
  createAgentSessionServices,
  createCodemodeExtension,
  type ExtensionContext,
  type ExtensionFactory,
  type ExtensionUIContext,
  FooterComponent,
  initTheme,
  ModelRuntime,
  type ReadonlyFooterDataProvider,
  SessionManager,
  SettingsManager,
  type Theme,
} from "@earendil-works/pi-coding-agent";
import { stripTerminalSequences } from "@earendil-works/pi-tui";
import { test as base, inject } from "vitest";

import { type DaemonStatus, type ResponseFor, send } from "../extensions/rbtr/daemon-client.js";
import type { ErrorResponse, Request, Response } from "../extensions/rbtr/generated/protocol.js";
import rbtrIndexExtension from "../extensions/rbtr/index.js";
import { RbtrReplyError } from "../extensions/rbtr/reply-error.js";
import type { Corpus } from "./global-setup.js";
import { type FakeDaemonHandle, startFakeDaemon } from "./helpers/fake-daemon.js";
import { TEST_RBTR_ENV } from "./rbtr-env.js";

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures");

/** A `command` setting for an rbtr that runs but cannot reach its index (`fixtures/daemon-at.ts`). */
export const DAEMON_AT = join(FIXTURES, "daemon-at.ts");

/**
 * Run *executable* as `rbtr --json` with *args*, in *env*. The fixtures' own
 * calls are not recorded in `processes`.
 */
function runRbtr(executable: string, env: NodeJS.ProcessEnv, ...args: string[]): string {
  const { RECORDED_PROCESSES: _, ...unrecorded } = env;
  return execFileSync(executable, ["--json", ...args], { encoding: "utf8", env: unrecorded });
}

/**
 * How long a test waits for a condition: 20 s, which leaves 10 s of the
 * 30 s test timeout for the test's own steps. A daemon that indexes or
 * starts under load takes most of it.
 */
export const READY_MS = 20_000;

/** Resolves once *ready* holds, checking every 50 ms; rejects after `READY_MS`. */
export async function until(ready: () => boolean | Promise<boolean>): Promise<void> {
  for (const deadline = Date.now() + READY_MS; !(await ready()); ) {
    if (Date.now() > deadline) throw new Error(`not ready within ${READY_MS} ms`);
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

/** The text the model reads from *result*. */
export function textOf(result: AgentToolResult<unknown>): string {
  return result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

/** A tool call the faux model makes. */
export interface Call {
  name: string;
  args: JsonObject;
}

/**
 * A finished tool call, as pi reports it in `tool_execution_end` after every
 * `tool_result` handler ran: `content` is what the model reads,
 * `structuredContent` what a codemode script gets. `input` is the arguments
 * the tool ran with, after pi's `prepareArguments` and validation, as pi
 * gives them to `tool_call` handlers. `updates` are the partial results pi
 * showed while it ran, in order.
 */
export interface ToolEnd {
  name: string;
  input: Record<string, unknown>;
  isError: boolean;
  result: AgentToolResult<unknown>;
  updates: AgentToolResult<unknown>[];
}

/** What the recording UI saw. */
export interface UIRecord {
  notifications: string[];
  /** Every footer status set, in order, without terminal styling. */
  statuses: string[];
  /** Answers `confirm` gives, in order; `false` once they run out. */
  confirmAnswers: boolean[];
}

export interface PiSession {
  runtime: AgentSessionRuntime;
  faux: FauxProviderHandle;
  ui: UIRecord;
  /**
   * Bind the extension to the runtime's current session, as pi does at
   * startup and after a new session, fork or switch: `session_start` runs.
   */
  start(options?: { mode?: ExtensionContext["mode"] }): Promise<void>;
  /**
   * The model makes *calls*, one assistant turn each, then answers; returns
   * how each ended. A list of calls is one turn, whose calls pi runs in
   * parallel.
   */
  run(...calls: (Call | Call[])[]): Promise<ToolEnd[]>;
  /** pi's footer at *width* columns, without terminal styling. */
  footer(width: number): string[];
}

/**
 * The rbtr extension, an extension recording what each call ran with, and,
 * with *builtins*, the built-in extensions pi's CLI loads that rbtr's tools
 * meet: codemode, registered inactive as in the CLI.
 */
function extensionsRecordingInputs(builtins: boolean): {
  extensions: ExtensionFactory[];
  inputs: Map<string, Record<string, unknown>>;
} {
  const inputs = new Map<string, Record<string, unknown>>();
  const recordInputs: ExtensionFactory = (pi) => {
    pi.on("tool_call", (event) => {
      inputs.set(event.toolCallId, structuredClone(event.input));
    });
  };
  const extensions = [rbtrIndexExtension, recordInputs];
  if (builtins) extensions.push(createCodemodeExtension());
  return { extensions, inputs };
}

/**
 * Drive *runtime* through *faux*. With a *ui*, `start` binds it and
 * `footer` renders its statuses; without one, the session has pi's no-UI
 * context, as an SDK host that passes none.
 */
function piSession(
  runtime: AgentSessionRuntime,
  faux: FauxProviderHandle,
  inputs: Map<string, Record<string, unknown>>,
  ui?: { context: ExtensionUIContext; record: UIRecord; footerData: ReadonlyFooterDataProvider },
): PiSession {
  return {
    runtime,
    faux,
    ui: ui?.record ?? { notifications: [], statuses: [], confirmAnswers: [] },
    async start(options = {}) {
      await runtime.session.bindExtensions(
        ui ? { uiContext: ui.context, mode: options.mode ?? "tui" } : { mode: options.mode },
      );
    },
    async run(...calls) {
      const ends: ToolEnd[] = [];
      const updates = new Map<string, AgentToolResult<unknown>[]>();
      const unsubscribe = runtime.session.subscribe((event) => {
        if (event.type === "tool_execution_update" && event.parentToolCallId === undefined) {
          updates.set(event.toolCallId, [...(updates.get(event.toolCallId) ?? []), event.partialResult]);
        }
        if (event.type === "tool_execution_end" && event.parentToolCallId === undefined) {
          ends.push({
            name: event.toolName,
            input: inputs.get(event.toolCallId) ?? {},
            isError: event.isError,
            result: event.result,
            updates: updates.get(event.toolCallId) ?? [],
          });
        }
      });
      faux.setResponses([
        ...calls.map((turn) => fauxAssistantMessage([turn].flat().map((call) => fauxToolCall(call.name, call.args)))),
        fauxAssistantMessage(fauxText("done")),
      ]);
      try {
        await runtime.session.prompt("go");
      } finally {
        unsubscribe();
      }
      return ends;
    },
    footer(width) {
      if (!ui) throw new Error("a session with no UI has no footer");
      return new FooterComponent(runtime.session, ui.footerData).render(width).map(stripTerminalSequences);
    },
  };
}

/** pi's settings, as a host passes them to `SettingsManager.inMemory`. */
type PiSettings = NonNullable<Parameters<typeof SettingsManager.inMemory>[0]>;

/**
 * A pi session runtime in *cwd* with *extensions* loaded, *settings* set
 * and nothing discovered, driven by *faux*.
 */
async function sessionRuntime(
  cwd: string,
  agentDir: string,
  faux: FauxProviderHandle,
  extensions: ExtensionFactory[],
  { projectTrusted = true, settings = {} }: { projectTrusted?: boolean; settings?: PiSettings } = {},
): Promise<AgentSessionRuntime> {
  const modelRuntime = await ModelRuntime.create({
    authPath: join(agentDir, "auth.json"),
    modelsPath: join(agentDir, "models.json"),
  });
  modelRuntime.registerNativeProvider(faux.provider);
  // An SDK host decides trust here; pi's CLI asks the user or reads trust.json.
  const settingsManager = SettingsManager.inMemory(settings, { projectTrusted });
  return createAgentSessionRuntime(
    async ({ cwd: sessionCwd, sessionManager, sessionStartEvent }) => {
      const services = await createAgentSessionServices({
        cwd: sessionCwd,
        agentDir,
        settingsManager,
        modelRuntime,
        resourceLoaderOptions: {
          noExtensions: true,
          noSkills: true,
          noPromptTemplates: true,
          noThemes: true,
          noContextFiles: true,
          extensionFactories: extensions,
        },
      });
      const created = await createAgentSessionFromServices({
        services,
        sessionManager,
        sessionStartEvent,
        model: faux.getModel(),
      });
      return { ...created, services, diagnostics: services.diagnostics };
    },
    { cwd, agentDir, sessionManager: SessionManager.inMemory(cwd) },
  );
}

/** Ask a daemon *request*, getting its reply, or the error it answered with. */
export type Ask = <R extends Request>(request: R) => Promise<ResponseFor<R["kind"]> | ErrorResponse>;

export const test = base.extend<{
  temp: string;
  shared: { dirs: Record<string, string>; rpc: string; executable: string };
  corpus: Corpus & { pristine: string };
  repo: string;
  ask: Ask;
  ownDaemon: boolean;
  rbtr: { agentDir: string; dataDir: string; logDir: string; cli: (...args: string[]) => string };
  cli: (...args: string[]) => string;
  projectSettings: Record<string, unknown>;
  longBuild: boolean;
  writableRepo: string;
  cwd: string;
  outside: string;
  unhandledRejections: unknown[];
  runningDaemon: DaemonStatus;
  frozenDaemon: { freeze(): void; thaw(): void };
  fakeDaemon: FakeDaemonHandle;
  missing: ("python3" | "rbtr" | "uvx")[];
  processes: () => string[];
  settingsScope: "global" | "project";
  holdOn: string;
  heldProgram: string;
  heldRbtr: { release(): void };
  projectTrusted: boolean;
  piSettings: PiSettings;
  theme: Theme;
  pi: PiSession;
  piReady: PiSession;
  sdkHost: PiSession;
}>({
  // Where the fixtures' directories go: the run's root, which an
  // interrupted run's next run removes.
  temp: [inject("run").root, { scope: "file" }],

  // The run's shared daemon (`global-setup.ts`), for this file: its
  // `RBTR_*_DIR` settings and the test model (`rbtr-env.ts`) in the
  // environment, its RPC endpoint, and the `rbtr` `just test-ts` puts on PATH
  // (the workspace's `.venv`), found before any test puts stand-ins on PATH.
  shared: [
    async ({ temp: _ }, use) => {
      const { dirs, rpc } = inject("shared");
      const env = { ...dirs, ...TEST_RBTR_ENV };
      const saved = Object.fromEntries(Object.keys(env).map((key) => [key, process.env[key]]));
      Object.assign(process.env, env);
      const executable = execFileSync("sh", ["-c", "command -v rbtr"], { encoding: "utf8" }).trim();
      await use({ dirs, rpc, executable });
      for (const [key, value] of Object.entries(saved)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    },
    { scope: "file" },
  ],

  // What the tests use from the run's repository (`global-setup.ts`).
  corpus: [async ({ task: _ }, use) => use(inject("corpus")), { scope: "file" }],

  // The run's repository, indexed by the shared daemon. Tests read it and
  // never write to it: a test that does works on `writableRepo`.
  repo: [async ({ corpus }, use) => use(corpus.repo), { scope: "file" }],

  // Ask the shared daemon a request about any repository.
  ask: async ({ shared }, use) => {
    await use(async (request) => {
      try {
        return await send(request, { rpcEndpoint: shared.rpc });
      } catch (err) {
        if (err instanceof RbtrReplyError) return { kind: "error", code: err.code, message: err.message };
        throw err;
      }
    });
  },

  // Whether the test gets a daemon of its own instead of the shared one: one
  // it may stop, start or freeze, which is not running when the test starts.
  // Set it with `test.override`.
  ownDaemon: false,

  // The test's rbtr and pi agent directory. Its rbtr is the shared daemon's;
  // with `ownDaemon` it is isolated: a copy of the pristine index, which
  // holds `repo`, and a daemon stopped at teardown.
  rbtr: async ({ temp, shared, corpus, ownDaemon }, use) => {
    const agentDir = mkdtempSync(join(temp, "pi-agent-"));
    const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
    process.env.PI_CODING_AGENT_DIR = agentDir;
    const own = ownDaemon
      ? {
          RBTR_DATA_DIR: mkdtempSync(join(temp, "rbtr-d-")),
          RBTR_CONFIG_DIR: mkdtempSync(join(temp, "rbtr-c-")),
          RBTR_LOG_DIR: mkdtempSync(join(temp, "rbtr-l-")),
        }
      : null;
    if (own) {
      cpSync(corpus.pristine, own.RBTR_DATA_DIR, { recursive: true });
      appendFileSync(inject("run").daemons, `${own.RBTR_DATA_DIR}\n`);
      Object.assign(process.env, own);
    }
    // Fixed now, so a call from a test still running after teardown has
    // restored `process.env` goes to this test's rbtr, never the user's.
    const env = { ...process.env };
    const cli = (...args: string[]) => runRbtr(shared.executable, env, ...args);
    const dirs = own ?? shared.dirs;
    await use({ agentDir, dataDir: dirs.RBTR_DATA_DIR, logDir: dirs.RBTR_LOG_DIR, cli });
    if (own) {
      cli("daemon", "stop");
      Object.assign(process.env, shared.dirs);
    }
    if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
    for (const dir of [agentDir, ...Object.values(own ?? {})]) rmSync(dir, { recursive: true, force: true });
  },

  // The test's `rbtr` CLI, for what a test does outside pi; its processes
  // are not recorded in `processes`.
  cli: async ({ rbtr }, use) => use(rbtr.cli),

  // The rbtr settings `writableRepo`'s project settings hold. Set it with
  // `test.override`.
  projectSettings: {},

  // Whether `writableRepo` takes long enough to build for a test to watch
  // the build. Set it with `test.override`.
  longBuild: false,

  // A repository of the test's own, which rbtr has never indexed: the
  // subject's file and one of its users, at their paths in the run's
  // repository. With `longBuild`, it also holds copies of the long file,
  // about 40,000 lines in all, so its build takes over a second. Each copy
  // ends with its own number of blank lines and a comment naming this
  // repository: rbtr reuses the chunks of a file it has parsed before, in
  // any repository, so identical copies would build at once. `scratch`
  // points at its one commit. Its project settings hold `projectSettings`,
  // when there are any; git ignores them.
  writableRepo: async ({ temp, corpus, longBuild, projectSettings }, use) => {
    const dir = resolve(mkdtempSync(join(temp, "rbtr-writable-")));
    const git = (...args: string[]) => execFileSync("git", args, { cwd: dir, encoding: "utf8" });
    git("init", "-q", "-b", "main");
    git("config", "user.email", "test@example.com");
    git("config", "user.name", "test");
    appendFileSync(join(dir, ".git", "info", "exclude"), ".pi/\n");
    for (const file of [corpus.subject.file, corpus.subject.user]) {
      mkdirSync(join(dir, dirname(file)), { recursive: true });
      cpSync(join(corpus.repo, file), join(dir, file));
    }
    if (longBuild) {
      const long = readFileSync(join(corpus.repo, corpus.long), "utf8");
      mkdirSync(join(dir, "copies"));
      for (let i = 1; i <= Math.ceil(40_000 / long.split("\n").length); i++) {
        const copy = `${long}${"\n".repeat(i)}# ${basename(dir)}\n`;
        writeFileSync(join(dir, "copies", `copy_${i}${extname(corpus.long)}`), copy);
      }
    }
    git("add", ".");
    git("commit", "-qm", "Files from the run's repository");
    git("branch", "scratch");
    if (Object.keys(projectSettings).length > 0) {
      mkdirSync(join(dir, ".pi"));
      writeFileSync(join(dir, ".pi", "rbtr-index.json"), JSON.stringify(projectSettings));
    }
    await use(dir);
    rmSync(dir, { recursive: true, force: true, maxRetries: 5 });
  },

  // The directory `pi`'s session runs in: `repo`. Override it with
  // `test.override("cwd", …)`.
  cwd: async ({ repo }, use) => use(repo),

  // A directory outside the repository holding a copy of the subject's
  // user, so a search there finds what one in the repository would.
  outside: [
    async ({ temp, corpus }, use) => {
      const dir = mkdtempSync(join(temp, "rbtr-outside-"));
      cpSync(join(corpus.repo, corpus.subject.user), join(dir, basename(corpus.subject.user)));
      await use(dir);
      rmSync(dir, { recursive: true, force: true });
    },
    { scope: "file" },
  ],

  // The reasons of the promises rejected with no handler during the test.
  unhandledRejections: async ({ task: _ }, use) => {
    const reasons: unknown[] = [];
    const record = (reason: unknown) => reasons.push(reason);
    process.on("unhandledRejection", record);
    await use(reasons);
    process.off("unhandledRejection", record);
  },

  // The test's own daemon (`ownDaemon`), started: the status `rbtr daemon
  // start` printed.
  runningDaemon: async ({ rbtr, ownDaemon }, use) => {
    if (!ownDaemon) throw new Error("runningDaemon needs ownDaemon: the shared daemon is not the test's to stop");
    await use(JSON.parse(rbtr.cli("daemon", "start")) as DaemonStatus);
  },

  // The programs `processes` makes look uninstalled. Set it with
  // `test.override`.
  missing: [],

  // `runningDaemon`, which `freeze()` stops with SIGSTOP: alive, but
  // answering nothing until `thaw()`, which teardown also calls. Freezing is
  // the test's step, after whatever must still reach the daemon.
  frozenDaemon: async ({ runningDaemon }, use) => {
    const pid = runningDaemon.pid;
    if (pid == null) throw new Error("the daemon reported no pid");
    let frozen = false;
    const freeze = () => {
      process.kill(pid, "SIGSTOP");
      frozen = true;
    };
    const thaw = () => {
      if (frozen) process.kill(pid, "SIGCONT");
      frozen = false;
    };
    await use({ freeze, thaw });
    thaw();
  },

  // A `FakeDaemon` the session finds as its daemon: the extension's `command`
  // is `daemon-at`, whose `daemon status` names the fake's endpoints and a
  // live pid at the extension's version. The fake answers a status request
  // with the repository at HEAD, indexed, and any other request with an
  // `internal` error.
  fakeDaemon: async ({ rbtr, repo }, use) => {
    const { version } = createRequire(import.meta.url)("../package.json") as { version: string };
    const status = (request: Request): Response =>
      request.kind === "status"
        ? {
            kind: "status",
            repo_path: repo,
            indexed_refs: [{ sha: "a".repeat(40), names: ["HEAD"], total: 1, embedded: 1, repo_path: repo }],
            watched: [{ ref: "HEAD", sha: "a".repeat(40), indexed: true, repo_path: repo }],
          }
        : { kind: "error", code: "internal", message: `the fake daemon serves no ${request.kind}` };
    const fake = await startFakeDaemon({ reply: status });
    const settings = join(rbtr.agentDir, "rbtr-index.json");
    writeFileSync(settings, JSON.stringify({ command: DAEMON_AT }));
    const report: DaemonStatus = {
      running: true,
      pid: process.pid,
      rpc: fake.endpoint,
      pub: fake.pubEndpoint,
      version,
    };
    process.env.RBTR_DAEMON_STATUS = JSON.stringify(report);
    await use(fake);
    delete process.env.RBTR_DAEMON_STATUS;
    rmSync(settings);
    await fake[Symbol.asyncDispose]();
  },

  // The command lines of the `python3`, `rbtr` and `uvx` processes started
  // during the test, in order. Recording wrappers go first on PATH, then
  // stand-ins for the `missing` programs, which exit 127.
  processes: async ({ temp, rbtr: _, missing }, use) => {
    const dir = mkdtempSync(join(temp, "rbtr-wrappers-"));
    const absent = mkdtempSync(join(temp, "rbtr-missing-"));
    const log = join(dir, "log");
    writeFileSync(log, "");
    for (const name of ["python3", "rbtr", "uvx"]) symlinkSync(join(FIXTURES, "record-process.ts"), join(dir, name));
    for (const name of missing) symlinkSync(join(FIXTURES, "not-installed.ts"), join(absent, name));
    const saved = { PATH: process.env.PATH, RECORDED_PROCESSES: process.env.RECORDED_PROCESSES };
    process.env.PATH = [dir, absent, saved.PATH ?? ""].join(delimiter);
    process.env.RECORDED_PROCESSES = log;
    await use(() =>
      readFileSync(log, "utf8")
        .split("\n")
        .filter((line) => line !== ""),
    );
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    for (const path of [dir, absent]) rmSync(path, { recursive: true, force: true });
  },

  // Which `rbtr-index.json` `heldRbtr` writes: the user's, in the agent
  // directory, or the project's, in `cwd`, which must then be
  // `writableRepo`. Set it with `test.override`.
  settingsScope: "global",

  // The rbtr subcommand `heldRbtr` holds; every one when empty. Set it with
  // `test.override`.
  holdOn: "",

  // The program `heldRbtr` runs once released. Set it with `test.override`.
  heldProgram: "rbtr",

  // The extension's `command` is `held-rbtr`, which runs the `holdOn`
  // subcommand only once `release()` is called; with no `holdOn`, startup's
  // first process waits until then.
  heldRbtr: async ({ temp, rbtr, repo, cwd, settingsScope, holdOn, heldProgram }, use) => {
    if (settingsScope === "project" && !relative(repo, cwd).startsWith("..")) {
      throw new Error("project settings go in writableRepo, not the run's repository: override cwd");
    }
    const release = join(mkdtempSync(join(temp, "rbtr-release-")), "release");
    const dir = settingsScope === "global" ? rbtr.agentDir : join(cwd, ".pi");
    mkdirSync(dir, { recursive: true });
    const settings = join(dir, "rbtr-index.json");
    writeFileSync(settings, JSON.stringify({ command: join(FIXTURES, "held-rbtr.ts") }));
    process.env.RBTR_RELEASE = release;
    process.env.RBTR_HOLD_ON = holdOn;
    process.env.RBTR_HELD_PROGRAM = heldProgram;
    await use({ release: () => writeFileSync(release, "") });
    writeFileSync(release, "");
    rmSync(settingsScope === "global" ? settings : dir, { recursive: true, force: true });
    delete process.env.RBTR_RELEASE;
    delete process.env.RBTR_HOLD_ON;
    delete process.env.RBTR_HELD_PROGRAM;
  },

  // Whether the session's host trusts the project. Set it with `test.override`.
  projectTrusted: true,

  // pi's settings for the session, as `settings.json` would hold them. Set
  // it with `test.override`.
  piSettings: {},

  // pi's theme, as pi gives it to an extension in a session with no UI.
  theme: [
    async ({ temp }, use) => {
      const agentDir = mkdtempSync(join(temp, "pi-agent-"));
      initTheme();
      let seen: Theme | undefined;
      const probe: ExtensionFactory = (pi) => {
        pi.on("session_start", (_event, ctx) => {
          seen = ctx.ui.theme;
        });
      };
      const runtime = await sessionRuntime(temp, agentDir, fauxProvider(), [probe]);
      await runtime.session.bindExtensions({});
      await runtime.dispose();
      rmSync(agentDir, { recursive: true, force: true });
      if (seen === undefined) throw new Error("pi gave the probe extension no theme");
      await use(seen);
    },
    { scope: "file" },
  ],

  pi: async ({ rbtr, cwd, projectTrusted, piSettings, theme }, use) => {
    const faux = fauxProvider();
    const { extensions, inputs } = extensionsRecordingInputs(true);
    const runtime = await sessionRuntime(cwd, rbtr.agentDir, faux, extensions, {
      projectTrusted,
      settings: piSettings,
    });
    // The footer's data: our statuses, no git branch. pi's FooterComponent
    // renders it.
    const statuses = new Map<string, string>();
    const footerData: ReadonlyFooterDataProvider = {
      getGitBranch: () => null,
      getExtensionStatuses: () => statuses,
      getAvailableProviderCount: () => 1,
      onBranchChange: () => () => {},
    };
    const record: UIRecord = { notifications: [], statuses: [], confirmAnswers: [] };
    const context: ExtensionUIContext = {
      select: async () => undefined,
      confirm: async () => record.confirmAnswers.shift() ?? false,
      input: async () => undefined,
      notify: (message) => record.notifications.push(message),
      onTerminalInput: () => () => {},
      setStatus: (key, text) => {
        if (text === undefined) statuses.delete(key);
        else statuses.set(key, text);
        if (text !== undefined) record.statuses.push(stripTerminalSequences(text));
      },
      setWorkingMessage: () => {},
      setWorkingVisible: () => {},
      setWorkingIndicator: () => {},
      setHiddenThinkingLabel: () => {},
      setWidget: () => {},
      setFooter: () => {},
      setHeader: () => {},
      setTitle: () => {},
      custom: async () => {
        throw new Error("the recording UI draws no custom components");
      },
      pasteToEditor: () => {},
      setEditorText: () => {},
      getEditorText: () => "",
      editor: async () => undefined,
      addAutocompleteProvider: () => {},
      setEditorComponent: () => {},
      getEditorComponent: () => undefined,
      theme,
      getAllThemes: () => [],
      getTheme: () => undefined,
      setTheme: () => ({ success: false, error: "the recording UI has no themes" }),
      getToolsExpanded: () => false,
      setToolsExpanded: () => {},
    };
    await use(piSession(runtime, faux, inputs, { context, record, footerData }));
    await runtime.dispose();
  },

  // `pi`, started, once its startup has finished: the footer shows the index.
  piReady: async ({ pi }, use) => {
    await pi.start();
    await until(() => / symbols/.test(pi.ui.statuses.at(-1) ?? ""));
    await use(pi);
  },

  // A session whose host passes no UI and never initialises pi's theme, as
  // pi's SDK examples do. A file that uses it must not use `pi` or `theme`,
  // which initialise the theme for the whole file.
  sdkHost: async ({ rbtr, repo }, use) => {
    const faux = fauxProvider();
    const { extensions, inputs } = extensionsRecordingInputs(false);
    const runtime = await sessionRuntime(repo, rbtr.agentDir, faux, extensions);
    await use(piSession(runtime, faux, inputs));
    await runtime.dispose();
  },
});
