/**
 * rbtr — pi extension for the rbtr structural code index.
 *
 * Gives the LLM access to rbtr's code index via registered tools.
 * Tries the ZMQ daemon first; falls back to shelling the CLI on
 * transport errors.  Protocol types come from the generated
 * `./generated/protocol.ts` — Python is the source of truth.
 */

import { readFile, realpath } from "node:fs/promises";
import { createRequire } from "node:module";
import { isAbsolute, relative, resolve } from "node:path";
import type {
  AgentToolResult,
  AgentToolUpdateCallback,
  ExtensionAPI,
  ExtensionContext,
  ToolAnnotations,
  ToolNamespace,
} from "@earendil-works/pi-coding-agent";
import { getSettingsListTheme, isBashToolResult, isReadToolResult } from "@earendil-works/pi-coding-agent";
import { Container, type SettingItem, SettingsList } from "@earendil-works/pi-tui";
import { Type } from "typebox";

import { classifyDaemonFailure, decideStartupDecision } from "./classify.js";
import type { ResponseFor } from "./daemon-client.js";
import { DaemonSession, DaemonUnavailableError, type ReconcileResult } from "./daemon-session.js";
import { errorReply, type ResolvedCommand, resolveCommand, runRbtrJson, statusReport } from "./exec.js";
import { Footer } from "./footer.js";
import type {
  ChangedSymbolsResponse,
  FindRefsResponse,
  GcMode,
  GcResponse,
  ListSymbolsResponse,
  ProgressNotification,
  ReadSymbolResponse,
  Request,
  Response,
  SearchResponse,
  StatusResponse,
  UnwatchResponse,
  WatchResponse,
  WatchSetResponse,
} from "./generated/protocol.js";
import { RbtrReplyError } from "./reply-error.js";

const require = createRequire(import.meta.url);
const { version: EXTENSION_VERSION } = require("../../package.json") as { version: string };

/**
 * How long a CLI fallback may take for a read tool.
 *
 * Must exceed the CLI client's own wait budget: the daemon it
 * talks to may be indexing, and killing the process at a shorter
 * deadline means the waiting the client would have done never
 * happens.
 */
const READ_CLI_TIMEOUT_MS = 150_000;

/** How long a read waits for a build of what it asks for before giving rbtr's "in progress" reply. */
const BUILD_WAIT_MS = 30_000;

/** A read's partial result while it waits for a build: the build's phase, once one is reported. */
function waitingFor(progress: ProgressNotification | null): AgentToolResult<unknown> {
  const text =
    progress === null
      ? "waiting for the build"
      : `waiting for the build: ${progress.phase} ${progress.current}/${progress.total}`;
  return { content: [{ type: "text", text }], details: { waiting: progress } };
}

/** How often the health check looks for a daemon that died or came back. */
export const HEALTH_CHECK_MS = 30_000;

import { block, type IndexLookups, outlinedIn, readFacts, searchesOutside, searchFacts } from "./annotate.js";
import { commandRefs, prepareStringLists } from "./args.js";
import { replySchema } from "./output-schema.js";
import {
  footerLabel,
  formatElapsed,
  renderChangedSymbolsCall,
  renderChangedSymbolsResult,
  renderFindRefsCall,
  renderFindRefsResult,
  renderIndexCall,
  renderIndexResult,
  renderListSymbolsCall,
  renderListSymbolsResult,
  renderReadSymbolCall,
  renderReadSymbolResult,
  renderSearchCall,
  renderSearchResult,
  renderStatusCall,
  renderStatusResult,
} from "./render.js";
import { loadSettings, type RbtrIndexSettings, saveProjectSettings } from "./settings.js";
import { LOADER, ON_REQUEST, withOnRequest } from "./tool-set.js";

// ── Tool result shape ─────────────────────────────────────────

type ToolReturn = AgentToolResult<Record<string, unknown>>;

/**
 * Pack an rbtr response: its JSON for the model, the same value as the
 * structured result for scripts, and the response for the renderer.
 */
function toolResult(response: Response): ToolReturn {
  const text = JSON.stringify(response);
  return { content: [{ type: "text", text }], structuredContent: JSON.parse(text), details: { response } };
}

// ── Tool metadata ─────────────────────────────────────────────

/** The group every rbtr tool belongs to, for codemode listings and tool search. */
const NAMESPACE: ToolNamespace = {
  name: "rbtr",
  description:
    "rbtr's structural code index of this repository: search by meaning, read symbols, outlines, dependency edges, symbol diffs.",
};

// Hints with the meaning of MCP tool annotations, for permission
// extensions deciding which calls to confirm. Every rbtr tool talks
// to the local index only.
const READS: ToolAnnotations = { readOnlyHint: true, openWorldHint: false };
/** Changes what the daemon indexes; deletes no data; repeating it changes nothing more. */
const WATCHES: ToolAnnotations = {
  readOnlyHint: false,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: false,
};
const DELETES: ToolAnnotations = {
  readOnlyHint: false,
  destructiveHint: true,
  idempotentHint: false,
  openWorldHint: false,
};

// ── Extension ─────────────────────────────────────────────────

/**
 * Surface reconcile outcomes to the user.  `up_to_date` and
 * `older_client` are silent — they're the common/boring cases.
 * `started`, `restarted`, and `failed` deserve a notification so
 * the user understands what just happened (especially "newer
 * extension restarted your daemon; an in-flight build was
 * killed", which is the intended semantics of the restart path).
 */
function notifyReconcile(ctx: ExtensionContext, result: ReconcileResult): void {
  switch (result.outcome) {
    case "started":
      ctx.ui.notify(`rbtr daemon started (v${result.newVersion ?? "?"})`, "info");
      break;
    case "restarted":
      ctx.ui.notify(
        `rbtr daemon restarted: v${result.previousVersion ?? "?"} → v${result.newVersion ?? "?"}. ` +
          `Any in-flight build was killed; the watcher will re-queue it.`,
        "info",
      );
      break;
    case "failed": {
      const why = result.detail ? `: ${result.detail}` : "";
      ctx.ui.notify(`rbtr daemon start/restart failed${why}. Falling back to CLI mode.`, "warning");
      break;
    }
    case "older_client":
    case "up_to_date":
      // silent — normal operation
      break;
  }
}

export default function rbtrIndexExtension(pi: ExtensionAPI) {
  const session = new DaemonSession();
  let resolved: ResolvedCommand | null = null;
  let settings: RbtrIndexSettings = { command: "rbtr", autoIndex: true, annotate: true };
  // What startup found out about the CLI; "unknown" until it has.
  let cli: "unknown" | "available" | "missing" = "unknown";
  let footer: Footer | null = null;
  let healthTimer: ReturnType<typeof setInterval> | null = null;
  // The repository's root as rbtr names it, from its status and watch
  // replies (ARCHITECTURE: "The repository root comes from rbtr").
  let repoRoot: string | null = null;
  // Bumped on every session start and shutdown, so a startup that
  // finishes after its session ended changes nothing.
  let generation = 0;
  // Settles when this session's startup has decided how tools reach
  // rbtr. Never rejects.
  let startup: Promise<void> = Promise.resolve();
  // The files a read of this turn outlines. pi runs a turn's calls in
  // parallel, so no read's result is in the context while another's is
  // noted; by the next turn's start they all are.
  const outlinedThisTurn = new Set<string>();

  /** Wait for startup, unless *signal* aborts first. */
  async function untilStarted(signal: AbortSignal | undefined): Promise<void> {
    if (signal === undefined) return startup;
    signal.throwIfAborted();
    await new Promise<void>((resolve, reject) => {
      const onAbort = () => reject(signal.reason);
      signal.addEventListener("abort", onAbort, { once: true });
      void startup.then(() => {
        signal.removeEventListener("abort", onAbort);
        resolve();
      });
    });
  }

  /**
   * Wait for startup, then try the daemon path, falling back to the
   * CLI callback on transport failure.  Propagates
   * ``RbtrReplyError`` from the daemon untouched — that is an
   * actionable reply, not a transport problem.
   */
  /**
   * Send a read to the daemon. When rbtr replies that the index is building
   * what the read asks for, wait for the build to end, at most
   * `BUILD_WAIT_MS`, showing its progress through *onUpdate*, then ask once
   * more; a second such reply, or no end in time, is rbtr's reply as given.
   * The listener goes on before the request, so a build that ends between
   * the reply and the wait is not missed.
   */
  async function sendWaitingOutBuild<R extends Request>(
    request: R,
    signal: AbortSignal | undefined,
    onUpdate: AgentToolUpdateCallback | undefined,
  ): Promise<ResponseFor<R["kind"]>> {
    let waiting = false;
    let ended = false;
    let wake = () => {};
    const stop = session.onNotification((notification) => {
      if (notification.repo_path !== repoRoot) return;
      if (notification.kind === "progress" && waiting) onUpdate?.(waitingFor(notification));
      if (notification.kind === "ready" || notification.kind === "index_error") {
        ended = true;
        wake();
      }
    });
    try {
      return await session.send(request, signal);
    } catch (err) {
      if (!(err instanceof RbtrReplyError) || err.code !== "index_in_progress") throw err;
      waiting = true;
      onUpdate?.(waitingFor(null));
      const buildEnded = await new Promise<boolean>((resolve, reject) => {
        if (ended) return resolve(true);
        const timer = setTimeout(() => resolve(false), BUILD_WAIT_MS);
        wake = () => {
          clearTimeout(timer);
          resolve(true);
        };
        signal?.addEventListener(
          "abort",
          () => {
            clearTimeout(timer);
            reject(signal.reason);
          },
          { once: true },
        );
      });
      if (!buildEnded) throw err;
      return await session.send(request, signal);
    } finally {
      stop();
    }
  }

  async function withFallback<T>(
    fromDaemon: () => Promise<T>,
    fromCli: () => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    await untilStarted(signal);
    if (session.available) {
      try {
        return await fromDaemon();
      } catch (err) {
        if (err instanceof DaemonUnavailableError) {
          // Transport failure — fall through to CLI.
        } else {
          throw err;
        }
      }
    }
    if (!resolved || cli !== "available") {
      throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
    }
    return fromCli();
  }

  function mapDaemonError(err: unknown): ToolReturn {
    if (err instanceof RbtrReplyError) {
      return { ...toolResult({ kind: "error", code: err.code, message: err.message }), isError: true };
    }
    throw err;
  }

  async function queryIndexStatus(repo: string): Promise<StatusResponse | null> {
    const status = await indexStatus(repo);
    if (status) learnRepoRoot(status);
    return status;
  }

  async function indexStatus(repo: string): Promise<StatusResponse | null> {
    if (session.available) {
      try {
        return await session.send({ kind: "status", repo_path: repo });
      } catch (err) {
        if (err instanceof RbtrReplyError) return null;
        // transport — fall through to CLI
      }
    }
    if (!resolved) return null;
    try {
      return await runRbtrJson<StatusResponse>(pi, resolved, ["status"], { timeout: 5000 });
    } catch {
      return null;
    }
  }

  /** Take the repository's root from a reply: a status states it; a watch reply names it on each ref. */
  function learnRepoRoot(reply: StatusResponse | WatchSetResponse | WatchResponse): void {
    const named =
      reply.kind === "status"
        ? reply.repo_path
        : "watched" in reply
          ? reply.watched?.find((entry) => entry.repo_path)?.repo_path
          : null;
    repoRoot = named ?? repoRoot;
  }

  // ── Index facts in bash and read output ────────────────────

  function lookups(cwd: string, signal: AbortSignal | undefined): IndexLookups {
    return {
      readSymbol: (symbol) => session.send({ kind: "read_symbol", repo_path: cwd, symbol }, signal),
      findRefs: (symbol) => session.send({ kind: "find_refs", repo_path: cwd, symbol }, signal),
      search: (query) => session.send({ kind: "search", repo_path: cwd, query, limit: 3 }, signal),
      listSymbols: (file_path) => session.send({ kind: "list_symbols", repo_path: cwd, file_path }, signal),
    };
  }

  // Daemon only: a CLI fallback would start a process per annotated call.
  // Any failure leaves the tool's output as it was.
  pi.on("tool_result", async (event, ctx) => {
    // A call another tool made (a codemode script) returns to that tool,
    // not to the model.
    if (!settings.annotate || event.isError || event.parentToolCallId) return;
    let lines: string[] = [];
    try {
      await untilStarted(ctx.signal);
      if (!session.available) return;
      // Real paths, as rbtr names the root; the working directory until
      // rbtr has named it.
      const cwd = await realpath(ctx.cwd);
      const root = repoRoot ?? cwd;
      const { command, path, offset } = event.input;
      if (isBashToolResult(event) && typeof command === "string") {
        // The index knows this repository only.
        if (searchesOutside(command, cwd, root)) return;
        const output = event.content.map((part) => (part.type === "text" ? part.text : "")).join("");
        lines = await searchFacts(lookups(ctx.cwd, ctx.signal), command, output);
      } else if (isReadToolResult(event) && typeof path === "string" && offset === undefined) {
        const absolute = await realpath(resolve(cwd, path));
        const filePath = relative(root, absolute);
        if (filePath.startsWith("..") || isAbsolute(filePath)) return;
        // The model still has an outline that is in its context, or gets
        // one from another read of this turn.
        if (outlinedThisTurn.has(filePath) || outlinedIn(ctx.sessionManager.buildContextEntries(), filePath)) return;
        outlinedThisTurn.add(filePath);
        const lineCount =
          event.details?.truncation?.totalLines ?? (await readFile(absolute, "utf8")).split("\n").length;
        lines = await readFacts(lookups(ctx.cwd, ctx.signal), filePath, lineCount);
      }
    } catch {
      return;
    }
    const text = block(lines);
    if (text === null) return;
    // pi drops a tool's details and structured result unless the handler returns them.
    return {
      content: [...event.content, { type: "text", text: `\n\n${text}` }],
      details: event.details,
      structuredContent: event.structuredContent,
    };
  });

  pi.on("turn_start", () => {
    outlinedThisTurn.clear();
  });

  // ── Session lifecycle ───────────────────────────────────────

  pi.on("session_start", (_event, ctx) => {
    settings = loadSettings(ctx.cwd, ctx.isProjectTrusted());
    repoRoot = null;
    footer = new Footer(ctx);
    footer.setSpinner("muted", (frame) => `rbtr: ${frame} starting…`);
    const current = ++generation;
    // Startup runs in the background; tools wait for `startup` themselves.
    startup = start(ctx, () => current === generation).catch((err) => {
      if (current !== generation) return;
      footer?.setStatic("error", "rbtr: ✗ error");
      ctx.ui.notify(`rbtr failed to start: ${err instanceof Error ? err.message : String(err)}`, "error");
    });
  });

  /**
   * Find, start and probe the daemon, then index if the repository
   * has no index.  *isCurrent* turns false once the session has
   * ended; nothing past that point touches the session.
   */
  async function start(ctx: ExtensionContext, isCurrent: () => boolean): Promise<void> {
    // Finding rbtr also reads its daemon's status. Without a daemon, the
    // tools run the CLI.
    const found = await resolveCommand(pi, settings.command);
    if (!isCurrent()) return;
    if (found.kind === "failed") {
      // Why rbtr did not run is unknown here; not being installed is the
      // likeliest cause, so the hint says so.
      cli = "missing";
      footer?.setStatic("error", "rbtr: cannot run");
      ctx.ui.notify(
        `rbtr could not be run (${found.command.description}: ${found.reason}). ` +
          "It may not be installed: uv tool install rbtr\n" +
          'Or set command in .pi/rbtr-index.json to "uvx" or "uvx --from <path>"',
        "warning",
      );
      return;
    }
    resolved = found.command;
    session.useCommand(found.command, found.daemon);

    // Reconcile daemon version with the extension: start if
    // missing, restart if we're newer, yield if older.  See
    // DaemonSession.reconcile for the full rule set.
    if (resolved) {
      const { executable, baseArgs } = found.command;
      const execDaemon = async (sub: "start" | "stop"): Promise<string> => {
        const result = await pi.exec(executable, [...baseArgs, "daemon", sub], { timeout: 15_000 });
        if (result.code !== 0) {
          // Under --json rbtr prints its error on stdout; a crash leaves only stderr.
          const failure = classifyDaemonFailure(result.code, errorReply(result.stdout), result.stderr ?? "");
          throw new Error(failure.message);
        }
        return result.stdout;
      };
      const reconcileResult = await session.reconcile(EXTENSION_VERSION, {
        startDaemon: async () => {
          footer?.setSpinner("muted", (frame) => `rbtr: ${frame} starting daemon…`);
          return statusReport(await execDaemon("start"));
        },
        stopDaemon: async () => {
          footer?.setSpinner("muted", (frame) => `rbtr: ${frame} restarting daemon…`);
          await execDaemon("stop");
        },
      });
      if (!isCurrent()) return;
      notifyReconcile(ctx, reconcileResult);
    }

    if (session.available) {
      startSubscription(ctx);
    }

    healthTimer = setInterval(() => {
      void checkDaemonHealth(ctx, isCurrent);
    }, HEALTH_CHECK_MS);

    const status = await queryIndexStatus(ctx.cwd);
    if (!isCurrent()) return;
    const decision = decideStartupDecision(status, settings.autoIndex);

    cli = "available";

    if (decision.kind === "indexed") {
      const top = (status?.indexed_refs ?? [])[0];
      if (top) footer?.setStatic("success", footerLabel(top, session.available));
      return;
    }

    if (decision.kind === "transient") {
      ctx.ui.notify("rbtr index temporarily unavailable (database busy); will retry.", "warning");
    }

    // empty or transient: index if enabled (the CLI auto-starts /
    // falls back), else leave a hint in the footer.  Startup does not
    // wait for the build: the footer reports it.
    if (decision.index) {
      void triggerIndex(ctx);
    } else {
      const hint =
        decision.kind === "transient" ? "rbtr: index unavailable — retrying" : "rbtr: no index — /rbtr-index";
      footer?.setStatic("muted", hint);
    }
  }

  pi.on("session_shutdown", async () => {
    generation++;
    if (healthTimer !== null) {
      clearInterval(healthTimer);
      healthTimer = null;
    }
    session.stopSubscribing();
    footer?.dispose();
    footer = null;
  });

  /**
   * Periodic health check: detect daemon death/return and update
   * the footer accordingly.  The ZMQ SUB socket auto-reconnects
   * on the transport layer; this function handles the UI gap.
   * *isCurrent* turns false once the session has ended; nothing past
   * that point touches the session.
   */
  async function checkDaemonHealth(ctx: ExtensionContext, isCurrent: () => boolean): Promise<void> {
    const transition = await session.detectTransition();
    if (!isCurrent()) return;
    switch (transition.kind) {
      case "died": {
        const status = await queryIndexStatus(ctx.cwd);
        const indexed = status?.indexed_refs ?? [];
        if (indexed.length > 0) {
          footer?.setStatic("success", footerLabel(indexed[0], false));
        } else {
          footer?.setStatic("muted", "rbtr: no index · no daemon");
        }
        break;
      }
      case "returned": {
        const status = await queryIndexStatus(ctx.cwd);
        if (!isCurrent()) return;
        const indexed = status?.indexed_refs ?? [];
        if (indexed.length > 0) {
          footer?.setStatic("success", footerLabel(indexed[0], true));
        } else {
          footer?.setStatic("muted", "rbtr: no index");
        }
        startSubscription(ctx);
        break;
      }
      case "unchanged":
        break;
    }
  }

  /**
   * Subscribe to daemon notifications and drive the footer off
   * them.  Keeps only those for this session's repository, by the
   * root rbtr named in its replies (`repoRoot`), so traffic for
   * other repos the daemon watches is ignored.
   */
  function startSubscription(ctx: ExtensionContext): void {
    let buildStartedAt: number | null = null;

    const elapsedSuffix = (): string => {
      if (buildStartedAt === null) return "";
      return ` · ${formatElapsed(Math.floor((Date.now() - buildStartedAt) / 1000))}`;
    };

    try {
      session.subscribe((notification) => {
        if (notification.repo_path !== repoRoot) return;
        if (!footer) return;
        switch (notification.kind) {
          case "progress": {
            const { phase, current, total } = notification;
            if (buildStartedAt === null) buildStartedAt = Date.now();
            footer.setSpinner("muted", (frame) =>
              total > 0
                ? `rbtr: ${frame} ${phase} ${current}/${total}${elapsedSuffix()}…`
                : `rbtr: ${frame} ${phase}${elapsedSuffix()}…`,
            );
            break;
          }
          case "ready":
            buildStartedAt = null;
            footer.setStatic(
              "success",
              footerLabel({ total: notification.chunks, embedded: notification.embedded }, true),
            );
            break;
          case "embed_ended":
            // An embed run that stood aside for a build, or stopped on
            // shutdown, leaves chunks unembedded and will run again —
            // only a finished one has nothing left to do.
            buildStartedAt = null;
            footer.setStatic(
              notification.outcome === "finished" ? "success" : "muted",
              footerLabel({ total: notification.chunks, embedded: notification.embedded }, true),
            );
            break;
          case "auto_rebuild":
            buildStartedAt = Date.now();
            footer.setSpinner("muted", (frame) => `rbtr: ${frame} rebuilding${elapsedSuffix()}…`);
            break;
          case "index_error":
            buildStartedAt = null;
            footer.setStatic("error", "rbtr: ✗ error");
            ctx.ui.notify(notification.message, "error");
            break;
        }
      });
    } catch {
      // PUB subscription is best-effort; a failure here doesn't
      // make the extension non-functional, just means the
      // footer won't update on its own.
    }
  }

  // pi's `rbtr` prompt section (ARCHITECTURE: "What the model reads first").
  pi.on("before_agent_start", async (event) => {
    const { sections } = event.systemPromptOptions;
    if (cli === "missing") {
      delete sections.rbtr;
      return;
    }
    sections.rbtr =
      "This repository has an rbtr code index. Use the rbtr_* tools to find code by meaning, structure or name; use grep for exact strings.";
  });

  /**
   * Watch refs (daemon first, CLI fallback) and return rbtr's reply.
   *
   * The daemon replies with the watch set at once and builds in the
   * background; progress arrives via PUB notifications.  The CLI's
   * inline build replies with what it built.  A failure marks the
   * footer and propagates.
   */
  async function watchRefs(
    ctx: ExtensionContext,
    refs: readonly string[],
    signal?: AbortSignal,
  ): Promise<WatchSetResponse | WatchResponse> {
    const targetRefs = refs.length > 0 ? [...refs] : ["HEAD"];
    footer?.setSpinner("muted", (frame) => `rbtr: ${frame} indexing…`);
    try {
      const reply = await withFallback<WatchSetResponse | WatchResponse>(
        // The extension always builds the 'full' variant (the default).
        // The 'stripped' variant is benchmark-only, driven by rbtr-eval.
        () => session.send({ kind: "watch", repo_path: ctx.cwd, refs: targetRefs }, signal),
        () => {
          if (!resolved) throw new Error("rbtr CLI not available");
          return runRbtrJson<WatchSetResponse | WatchResponse>(pi, resolved, ["watch", ...targetRefs], {
            signal,
            timeout: 600_000,
          });
        },
        signal,
      );
      learnRepoRoot(reply);
      return reply;
    } catch (err) {
      footer?.setStatic("error", "rbtr: indexing failed");
      throw err;
    }
  }

  /**
   * Watch refs for the session start and `/rbtr-index`, telling the user
   * if it fails while the session lasts: a build outlives its session.
   */
  async function triggerIndex(ctx: ExtensionContext, ...refs: string[]): Promise<void> {
    const current = generation;
    try {
      await watchRefs(ctx, refs);
    } catch (err) {
      if (current !== generation) return;
      ctx.ui.notify(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }

  // Stop watching the given refs (daemon path, CLI fallback).  The
  // protocol takes at least one ref: unwatching nothing is a mis-shaped
  // call, and finding the stale ones is `removeStale`.  rbtr refuses HEAD.
  function unwatch(ctx: ExtensionContext, refs: [string, ...string[]], signal?: AbortSignal): Promise<UnwatchResponse> {
    return withFallback<UnwatchResponse>(
      () => session.send({ kind: "unwatch", repo_path: ctx.cwd, refs }, signal),
      () => {
        if (!resolved) throw new Error("rbtr CLI not available");
        return runRbtrJson<UnwatchResponse>(pi, resolved, ["unwatch", ...refs], { signal, timeout: 60_000 });
      },
      signal,
    );
  }

  // Drop watched refs that no longer resolve (e.g. deleted branches).
  // Which refs those are is rbtr's to decide, so this asks rather than
  // working it out here; HEAD is never removed.
  function removeStale(ctx: ExtensionContext, signal?: AbortSignal): Promise<UnwatchResponse> {
    return withFallback<UnwatchResponse>(
      () => session.send({ kind: "unwatch_stale", repo_path: ctx.cwd }, signal),
      () => {
        if (!resolved) throw new Error("rbtr CLI not available");
        return runRbtrJson<UnwatchResponse>(pi, resolved, ["unwatch", "--stale"], { signal, timeout: 60_000 });
      },
      signal,
    );
  }

  async function triggerGc(
    ctx: ExtensionContext,
    opts: { watchedOnly: boolean; dryRun: boolean },
    signal?: AbortSignal,
  ): Promise<GcResponse> {
    const mode: GcMode = opts.watchedOnly ? "watched_only" : "watched";
    return withFallback(
      async () => session.send({ kind: "gc", repo_path: ctx.cwd, mode, refs: [], dry_run: opts.dryRun }, signal),
      async () => {
        if (!resolved) throw new Error("rbtr CLI not available");
        const args = ["gc"];
        if (opts.watchedOnly) args.push("--keep", "watched-only");
        if (opts.dryRun) args.push("--dry-run");
        return runRbtrJson<GcResponse>(pi, resolved, args, { signal, timeout: 120_000 });
      },
      signal,
    );
  }

  // ── Commands ────────────────────────────────────────────────

  pi.registerCommand("rbtr-status", {
    description: "Show rbtr index status",
    handler: async (_args, ctx) => {
      await startup;
      if (cli !== "available") {
        ctx.ui.notify("rbtr CLI not available", "error");
        return;
      }
      const status = await queryIndexStatus(ctx.cwd);
      if (!status) {
        ctx.ui.notify("Failed to get index status", "error");
        return;
      }
      const refs = status.indexed_refs ?? [];
      if (refs.length > 0) {
        ctx.ui.notify(`Index: ${refs[0].total} symbols\nPath: ${status.db_path}`, "info");
      } else {
        ctx.ui.notify("No index found. Use /rbtr-index to create one.", "warning");
      }
    },
  });

  pi.registerCommand("rbtr-index", {
    description: "Index the repository, or the given refs: /rbtr-index main feature-x",
    handler: async (args, ctx) => {
      await startup;
      if (cli !== "available") {
        ctx.ui.notify("rbtr CLI not available", "error");
        return;
      }
      const refs = commandRefs(args);
      // A CLI build runs inline for minutes; the command does not wait for it.
      void triggerIndex(ctx, ...refs);
      ctx.ui.notify(`Indexing ${refs.length > 0 ? refs.join(", ") : "HEAD"}. Progress in the footer.`, "info");
    },
  });

  pi.registerCommand("rbtr-settings", {
    description: "View and toggle rbtr index settings",
    handler: async (_args, ctx) => {
      // A custom component needs pi's terminal UI; RPC has dialogs but no `custom()`.
      if (ctx.mode !== "tui") {
        ctx.ui.notify("/rbtr-settings requires the interactive terminal UI", "error");
        return;
      }
      if (!ctx.isProjectTrusted()) {
        ctx.ui.notify("Project settings are ignored until the project is trusted (/trust).", "warning");
        return;
      }

      await ctx.ui.custom((tui, theme, _kb, done) => {
        const items: SettingItem[] = [
          {
            id: "autoIndex",
            label: "Auto-index on session start",
            currentValue: settings.autoIndex ? "on" : "off",
            values: ["on", "off"],
          },
        ];

        const container = new Container();

        container.addChild({
          render(_width: number) {
            const desc = resolved ? resolved.description : "not resolved";
            const daemonMode = session.available ? "daemon" : "cli fallback";
            return [
              theme.fg("accent", theme.bold("rbtr Index Settings")),
              "",
              `${theme.fg("muted", "Command:")} ${settings.command}`,
              `${theme.fg("muted", "Resolved:")} ${theme.fg("dim", desc)}`,
              `${theme.fg("muted", "CLI available:")} ${cli === "available" ? theme.fg("success", "yes") : theme.fg("error", cli === "missing" ? "no" : "checking…")}`,
              `${theme.fg("muted", "Transport:")} ${theme.fg(session.available ? "success" : "warning", daemonMode)}`,
              "",
            ];
          },
          invalidate() {},
        });

        const settingsList = new SettingsList(
          items,
          Math.min(items.length + 2, 15),
          getSettingsListTheme(),
          (id, newValue) => {
            if (id === "autoIndex") {
              settings.autoIndex = newValue === "on";
              saveProjectSettings(ctx.cwd, { autoIndex: settings.autoIndex });
            }
          },
          () => done(undefined),
        );

        container.addChild(settingsList);

        return {
          render(width: number) {
            return container.render(width);
          },
          invalidate() {
            container.invalidate();
          },
          handleInput(data: string) {
            settingsList.handleInput?.(data);
            tui.requestRender();
          },
        };
      });
    },
  });

  // ── Tools ──────────────────────────────────────────────────

  pi.registerTool({
    name: "rbtr_search",
    label: "rbtr search",
    namespace: NAMESPACE,
    annotations: READS,
    description:
      "Find code by what it does or by name ('where are retries handled', 'the function that parses the config'): use it instead of grep when you don't know the exact text. It matches meaning: 'retry logic' also finds 'backoff' and 'reconnect'. Returns ranked functions, classes and methods with file, line and a preview of the source; a hit marked `clipped` has more lines, which rbtr_read_symbol returns. Pass `keywords` (synonyms) and, for a question in words, `variants` (rephrasings) to widen the match.",
    promptSnippet: "Find code by meaning or by name, when you don't know the exact text to grep for",
    promptGuidelines: [
      "Use rbtr_search instead of grep to find where something is handled or implemented; keep grep for exact strings such as error messages and config keys.",
      "Pass a hit's `name` to rbtr_read_symbol for its full source, and to rbtr_find_refs for what uses it.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      ref: Type.Optional(
        Type.String({
          description:
            "Git ref to read from (branch, tag, or SHA). Must be indexed. Defaults to the working tree if dirty, HEAD if clean.",
        }),
      ),
      limit: Type.Optional(Type.Number({ description: "Maximum results to return (default: 10)" })),
      keywords: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "3-5 keyword synonyms or alternative search terms to widen lexical matching. Omit for code fragments or exact identifiers.",
        }),
      ),
      variants: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "1-2 semantically diverse rephrases of the query for concept searches. Omit for identifiers and code.",
        }),
      ),
      scope: Type.Optional(
        Type.Union([Type.Literal("workspace"), Type.Literal("all")], {
          description: "Search breadth: 'workspace' (current repo, default) or 'all' (every indexed repo).",
        }),
      ),
    }),
    prepareArguments: prepareStringLists("keywords", "variants"),
    outputSchema: replySchema("SearchResponse", "ErrorResponse"),
    renderCall: (args, theme) => renderSearchCall(args, theme),
    renderResult: (result, options, theme) => renderSearchResult(result, options, theme),

    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      if (!params.query) throw new Error("Missing required parameter `query`. Example: {query: 'retry logic'}");
      try {
        const resp = await withFallback<SearchResponse>(
          () =>
            sendWaitingOutBuild(
              {
                kind: "search",
                repo_path: ctx.cwd,
                query: params.query,
                ...(params.ref !== undefined ? { ref: params.ref } : {}),
                ...(params.limit !== undefined ? { limit: params.limit } : {}),
                ...(params.keywords !== undefined ? { keywords: params.keywords } : {}),
                ...(params.variants !== undefined ? { variants: params.variants } : {}),
                ...(params.scope !== undefined ? { scope: params.scope } : {}),
              },
              signal,
              onUpdate,
            ),
          () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const args = ["search", params.query];
            if (params.ref !== undefined) args.push("--ref", params.ref);
            if (params.limit !== undefined) args.push("--limit", String(params.limit));
            if (params.scope !== undefined) args.push("--scope", params.scope);
            return runRbtrJson<SearchResponse>(pi, resolved, args, { signal, timeout: READ_CLI_TIMEOUT_MS });
          },
          signal,
        );
        return toolResult(resp);
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_read_symbol",
    label: "rbtr read-symbol",
    namespace: NAMESPACE,
    annotations: READS,
    description:
      "Read the full source of a function, class, method or constant by name, with no file path or line numbers: 'fuse_scores', 'HttpClient.retry'. Use it instead of grep then read whenever you know the name. Try the plain name first; qualify it ('module.Class.method') or pass `file_paths` when several symbols share it. 'Symbol not found' means the index has no such name at that ref: fall back to grep.",
    promptSnippet: "Read a function, class or method's source by name, instead of grep then read",
    promptGuidelines: [
      "Use rbtr_read_symbol instead of grep and read whenever you know the name of the function, class or method you want to see.",
    ],
    parameters: Type.Object({
      symbol: Type.String({
        description:
          "Symbol name as stored in the index. Examples: 'fuse_scores', 'HttpClient.retry', 'rbtr.index.search.fuse_scores'.",
      }),
      ref: Type.Optional(
        Type.String({
          description:
            "Git ref to read from (branch, tag, or SHA). Must be indexed. Defaults to the working tree if dirty, HEAD if clean.",
        }),
      ),
      file_paths: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "Restrict to symbols defined in these files. Use to disambiguate a name that collides across files.",
        }),
      ),
    }),
    prepareArguments: prepareStringLists("file_paths"),
    outputSchema: replySchema("ReadSymbolResponse", "ErrorResponse"),
    renderCall: (args, theme) => renderReadSymbolCall(args, theme),
    renderResult: (result, options, theme) => renderReadSymbolResult(result, options, theme),

    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      if (!params.symbol) throw new Error("Missing required parameter `symbol`. Example: {symbol: 'MyClass.method'}");
      try {
        const resp = await withFallback<ReadSymbolResponse>(
          () =>
            sendWaitingOutBuild(
              {
                kind: "read_symbol",
                repo_path: ctx.cwd,
                symbol: params.symbol,
                ...(params.ref !== undefined ? { ref: params.ref } : {}),
                ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
              },
              signal,
              onUpdate,
            ),
          () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const readArgs = ["read-symbol", params.symbol];
            if (params.ref !== undefined) readArgs.push("--ref", params.ref);
            for (const fp of params.file_paths ?? []) readArgs.push("--file-path", fp);
            return runRbtrJson<ReadSymbolResponse>(pi, resolved, readArgs, { signal, timeout: READ_CLI_TIMEOUT_MS });
          },
          signal,
        );
        return toolResult(resp);
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_find_refs",
    label: "rbtr find-refs",
    namespace: NAMESPACE,
    annotations: READS,
    description:
      "Find what imports or documents a symbol, from the index's dependency graph: the files and symbols that depend on it. Use it instead of grepping for a name when you need to know what a change would affect. Returns edges (source, target, kind `imports` or `docs`), not text matches; grep finds occurrences inside strings and comments.",
    promptSnippet: "Find what imports or documents a symbol, to see what a change would affect",
    promptGuidelines: [
      "Use rbtr_find_refs instead of grep before changing a function or class, to find the code that depends on it.",
    ],
    parameters: Type.Object({
      symbol: Type.String({
        description: "Symbol name (same format as rbtr_read_symbol: bare / class-qualified / module-qualified).",
      }),
      ref: Type.Optional(
        Type.String({
          description:
            "Git ref to read from (branch, tag, or SHA). Must be indexed. Defaults to the working tree if dirty, HEAD if clean.",
        }),
      ),
      file_paths: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "Restrict name resolution to symbols defined in these files. Use to disambiguate a name that collides across files.",
        }),
      ),
    }),
    prepareArguments: prepareStringLists("file_paths"),
    outputSchema: replySchema("FindRefsResponse", "ErrorResponse"),
    renderCall: (args, theme) => renderFindRefsCall(args, theme),
    renderResult: (result, options, theme) => renderFindRefsResult(result, options, theme),

    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      if (!params.symbol) throw new Error("Missing required parameter `symbol`. Example: {symbol: 'MyClass.method'}");
      try {
        const resp = await withFallback<FindRefsResponse>(
          () =>
            sendWaitingOutBuild(
              {
                kind: "find_refs",
                repo_path: ctx.cwd,
                symbol: params.symbol,
                ...(params.ref !== undefined ? { ref: params.ref } : {}),
                ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
              },
              signal,
              onUpdate,
            ),
          () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const findArgs = ["find-refs", params.symbol];
            if (params.ref !== undefined) findArgs.push("--ref", params.ref);
            for (const fp of params.file_paths ?? []) findArgs.push("--file-path", fp);
            return runRbtrJson<FindRefsResponse>(pi, resolved, findArgs, { signal, timeout: READ_CLI_TIMEOUT_MS });
          },
          signal,
        );
        return toolResult(resp);
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_changed_symbols",
    label: "rbtr changed-symbols",
    namespace: NAMESPACE,
    annotations: READS,
    description:
      "Diff two refs by symbol: which functions, classes and methods a branch added, changed or removed. Use it to review or summarise a branch instead of reading the whole line diff; use git diff for exact lines and for files that are not code. Both refs must be indexed: watch them first with rbtr_watch.",
    promptSnippet: "Which functions, classes and methods a branch added, changed or removed",
    promptGuidelines: [
      "Use rbtr_changed_symbols to review a branch or PR: call rbtr_watch with the branch and its base first, then diff them.",
    ],
    parameters: Type.Object({
      base: Type.String({ description: "Base ref (branch name, tag, or SHA). Must be indexed." }),
      head: Type.String({ description: "Head ref (branch name, tag, or SHA). Must be indexed." }),
      file_paths: Type.Optional(
        Type.Array(Type.String(), {
          description: "Scope the diff to these files. Only changes in the listed files are reported.",
        }),
      ),
    }),
    prepareArguments: prepareStringLists("file_paths"),
    outputSchema: replySchema("ChangedSymbolsResponse", "ErrorResponse"),
    renderCall: (args, theme) => renderChangedSymbolsCall(args, theme),
    renderResult: (result, options, theme) => renderChangedSymbolsResult(result, options, theme),

    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      if (!params.base || !params.head) {
        throw new Error(
          "Missing required parameters `base` and `head`. Example: {base: 'main', head: 'feature-branch'}",
        );
      }
      try {
        const resp = await withFallback<ChangedSymbolsResponse>(
          () =>
            sendWaitingOutBuild(
              {
                kind: "changed_symbols",
                repo_path: ctx.cwd,
                base: params.base,
                head: params.head,
                ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
              },
              signal,
              onUpdate,
            ),
          () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const changedArgs = ["changed-symbols", params.base, params.head];
            for (const fp of params.file_paths ?? []) changedArgs.push("--file-path", fp);
            return runRbtrJson<ChangedSymbolsResponse>(pi, resolved, changedArgs, {
              signal,
              timeout: READ_CLI_TIMEOUT_MS,
            });
          },
          signal,
        );
        return toolResult(resp);
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_list_symbols",
    label: "rbtr list-symbols",
    namespace: NAMESPACE,
    annotations: READS,
    description:
      "Outline one file: every function, class and method with its line range, without the source. Use it instead of reading a large file end to end to find the part you need, then read that range or call rbtr_read_symbol. Takes a `file` path relative to the repository root.",
    promptSnippet: "Outline a file's functions, classes and methods with their line ranges",
    promptGuidelines: ["Use rbtr_list_symbols on a large file before reading it, to find which part to read."],
    parameters: Type.Object({
      file: Type.String({ description: "File path relative to the repo root (e.g. 'src/rbtr/index/search.py')." }),
      ref: Type.Optional(
        Type.String({
          description:
            "Git ref to read from (branch, tag, or SHA). Must be indexed. Defaults to the working tree if dirty, HEAD if clean.",
        }),
      ),
    }),
    outputSchema: replySchema("ListSymbolsResponse", "ErrorResponse"),
    renderCall: (args, theme) => renderListSymbolsCall(args, theme),
    renderResult: (result, options, theme) => renderListSymbolsResult(result, options, theme),

    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      if (!params.file)
        throw new Error("Missing required parameter `file`. Example: {file: 'src/rbtr/index/search.py'}");
      try {
        const resp = await withFallback<ListSymbolsResponse>(
          () =>
            sendWaitingOutBuild(
              {
                kind: "list_symbols",
                repo_path: ctx.cwd,
                file_path: params.file,
                ...(params.ref !== undefined ? { ref: params.ref } : {}),
              },
              signal,
              onUpdate,
            ),
          () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const listArgs = ["list-symbols", params.file];
            if (params.ref !== undefined) listArgs.push("--ref", params.ref);
            return runRbtrJson<ListSymbolsResponse>(pi, resolved, listArgs, {
              signal,
              timeout: READ_CLI_TIMEOUT_MS,
            });
          },
          signal,
        );
        return toolResult(resp);
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_watch",
    label: "rbtr watch",
    namespace: NAMESPACE,
    annotations: WATCHES,
    description:
      "Index refs so the other rbtr tools can read them: a branch, tag or SHA, such as a PR branch and its base before a review. HEAD and the working tree are indexed automatically. Returns at once; indexing continues in the background, and rbtr_status (loaded by rbtr_index_tools) shows its progress. `remove` stops watching refs; `remove_stale` drops refs that no longer resolve. Safe to call repeatedly.",
    promptSnippet: "Index refs, such as a PR branch and its base, so the rbtr tools can read them",
    promptGuidelines: [
      "Use rbtr_watch when the user asks for refs to be indexed, or when an rbtr tool replies that a ref is not indexed.",
    ],
    parameters: Type.Object({
      refs: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "Refs to watch and index (default: ['HEAD']). Each ref is an independent watch target: " +
            'pass every ref as a separate array element, e.g. ["main", "HEAD"] — never a single ' +
            'space-joined string like "main HEAD".',
        }),
      ),
      remove: Type.Optional(Type.Boolean({ description: "Stop watching the given refs (HEAD cannot be removed)." })),
      remove_stale: Type.Optional(
        Type.Boolean({ description: "Stop watching refs that no longer resolve (e.g. deleted branches)." }),
      ),
    }),
    prepareArguments: prepareStringLists("refs"),
    outputSchema: replySchema("WatchSetResponse", "WatchResponse", "UnwatchResponse", "ErrorResponse"),
    renderCall: (args, theme) => renderIndexCall(args, theme),
    renderResult: (result, options, theme) => renderIndexResult(result, options, theme),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      // `prepareArguments` has already decoded a JSON-encoded list.
      const given = params.refs ?? [];
      const refs: [string, ...string[]] = given.length > 0 ? [given[0], ...given.slice(1)] : ["HEAD"];
      try {
        if (params.remove_stale) return toolResult(await removeStale(ctx, signal));
        if (params.remove) return toolResult(await unwatch(ctx, refs, signal));
        return toolResult(await watchRefs(ctx, refs, signal));
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: LOADER,
    label: "rbtr index tools",
    namespace: NAMESPACE,
    // It changes which tools the model has: the model's to call, not a script's.
    exposure: "model-only",
    annotations: READS,
    description:
      "Load the tools that inspect and clean up the rbtr index: rbtr_status (what is indexed, and whether a build is still running) and rbtr_gc (reclaim disk space; destructive, only on the user's request).",
    promptSnippet: "Load rbtr_status and rbtr_gc, to inspect the index or reclaim its disk space",
    promptGuidelines: [
      "Call rbtr_index_tools when you need to check whether indexing has finished, or the user asks to reclaim index space.",
    ],
    parameters: Type.Object({}),
    outputSchema: Type.Object(
      { kind: Type.Literal("tools_loaded"), tools: Type.Array(Type.String()) },
      { additionalProperties: false },
    ),

    async execute() {
      pi.setActiveTools(withOnRequest(pi.getActiveTools()));
      const loaded = { kind: "tools_loaded", tools: [...ON_REQUEST] };
      return { content: [{ type: "text", text: JSON.stringify(loaded) }], structuredContent: loaded, details: loaded };
    },
  });

  pi.registerTool({
    name: "rbtr_status",
    defaultActive: false,
    label: "rbtr status",
    namespace: NAMESPACE,
    annotations: READS,
    description:
      "Report the rbtr index for this repository: symbol totals, which commits are indexed, the build running now with its progress, and queued work. `scope: 'all'` covers every indexed repository.",
    promptSnippet: "Check what the rbtr index holds, and whether a build is still running",
    promptGuidelines: ["Use rbtr_status to check whether a ref you asked rbtr_watch to index has finished."],
    parameters: Type.Object({
      scope: Type.Optional(
        Type.Union([Type.Literal("workspace"), Type.Literal("all")], {
          description: "Status breadth: 'workspace' (current repo, default) or 'all' (every indexed repo).",
        }),
      ),
    }),
    outputSchema: replySchema("StatusResponse", "ErrorResponse"),
    renderCall: (args, theme) => renderStatusCall(args, theme),
    renderResult: (result, options, theme) => renderStatusResult(result, options, theme),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const scope = params.scope ?? "workspace";
      try {
        const status = await withFallback<StatusResponse>(
          () => session.send({ kind: "status", repo_path: ctx.cwd, scope }, signal),
          () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const args = scope === "all" ? ["status", "--scope", "all"] : ["status"];
            return runRbtrJson<StatusResponse>(pi, resolved, args, { timeout: 5000 });
          },
          signal,
        );
        return toolResult(status);
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_gc",
    label: "rbtr gc",
    namespace: NAMESPACE,
    defaultActive: false,
    // It asks the user before deleting: the model's to call, not a script's.
    exposure: "model-only",
    annotations: DELETES,
    description:
      "⚠ DESTRUCTIVE, IRREVERSIBLE. Permanently deletes indexed data (commits, chunks, embeddings) from the rbtr index; there is no undo. Only for when the user has explicitly asked to reclaim index space. Runs as a dry-run preview by default; with dry_run=false it shows the user what it would drop and deletes only if they confirm. By default keeps HEAD, all local branches and tags, and the watch set, dropping only unreferenced commits; watched_only also drops unwatched branches and tags. The daemon never deletes anything unless this is called.",
    promptSnippet: "Delete unreferenced index data to reclaim disk space (destructive; only on the user's request)",
    promptGuidelines: ["Use rbtr_gc only when the user explicitly asks to reclaim index space."],
    parameters: Type.Object({
      watched_only: Type.Optional(
        Type.Boolean({ description: "Keep only HEAD and watched refs (drop unwatched branches/tags)." }),
      ),
      dry_run: Type.Optional(
        Type.Boolean({
          description:
            "Preview without deleting. Defaults to true (safe); false asks the user to confirm, then deletes.",
        }),
      ),
    }),
    outputSchema: replySchema("GcResponse", "ErrorResponse"),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      // Dry-run by default: a careless call previews. Deleting asks the
      // user, with what a dry run would drop; without a UI it refuses.
      const watchedOnly = params.watched_only ?? false;
      const dryRun = params.dry_run ?? true;
      try {
        if (!dryRun) {
          if (!ctx.hasUI) {
            throw new Error("Deleting index data needs the user's confirmation; run `rbtr gc` in a terminal.");
          }
          const preview = await triggerGc(ctx, { watchedOnly, dryRun: true }, signal);
          const what = `${preview.snapshots_dropped} commits and ${preview.chunks_freed} chunks would be deleted, with no undo.`;
          if (!(await ctx.ui.confirm("Delete rbtr index data?", what, { signal }))) {
            throw new Error("The user declined deleting index data.");
          }
        }
        return toolResult(await triggerGc(ctx, { watchedOnly, dryRun }, signal));
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });
}
