/**
 * rbtr — pi extension for the rbtr structural code index.
 *
 * Gives the LLM access to rbtr's code index via registered tools.
 * Tries the ZMQ daemon first; falls back to shelling the CLI on
 * transport errors.  Protocol types come from the generated
 * `./generated/protocol.ts` — Python is the source of truth.
 *
 * Placement: .pi/extensions/rbtr/index.ts
 */

import { createRequire } from "node:module";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  getSettingsListTheme,
  truncateHead,
} from "@earendil-works/pi-coding-agent";
import { Container, type SettingItem, SettingsList } from "@earendil-works/pi-tui";
import { Type } from "typebox";

import { classifyDaemonFailure, decideStartupDecision } from "./classify.js";
import { RbtrDaemonError } from "./daemon-client.js";
import { DaemonSession, DaemonUnavailableError, type ReconcileResult } from "./daemon-session.js";
import { type ResolvedCommand, resolveCommand, runRbtr, runRbtrJson } from "./exec.js";
import { Footer } from "./footer.js";
import type {
  GcMode,
  GcResponse,
  Response,
  StatusResponse,
  UnwatchResponse,
  WatchResponse,
} from "./generated/protocol.js";

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

import { commandRefs, decodeStringList, echoArgs } from "./args.js";
import {
  footerLabel,
  formatElapsed,
  formatJobCounts,
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
  renderStatusText,
  shortSha,
} from "./render.js";
import { loadSettings, type RbtrIndexSettings, saveProjectSettings } from "./settings.js";
import { LOADER, ON_REQUEST, startingTools, withOnRequest } from "./tool-set.js";

// ── Tool result shape ─────────────────────────────────────────

interface ToolReturn {
  content: Array<{ type: "text"; text: string }>;
  details: Record<string, unknown>;
}

/** Pack a typed daemon response for the LLM + renderer. */
function toolResultFromDaemon(response: Response): ToolReturn {
  return {
    content: [{ type: "text", text: JSON.stringify(response) }],
    details: { fromDaemon: true, response },
  };
}

/** Pack raw CLI stdout for the LLM + renderer. */
function toolResultFromCli(stdout: string, extra: Record<string, unknown>): ToolReturn {
  const truncation = truncateHead(stdout, {
    maxLines: DEFAULT_MAX_LINES,
    maxBytes: DEFAULT_MAX_BYTES,
  });
  let content = truncation.content;
  if (truncation.truncated) {
    content +=
      `\n\n[Output truncated: showing ${truncation.outputLines} of ` +
      `${truncation.totalLines} lines. Use --limit or rbtr_read_symbol for details.]`;
  }
  return {
    content: [{ type: "text", text: content }],
    details: { fromCli: true, truncated: truncation.truncated, ...extra },
  };
}

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
  let settings: RbtrIndexSettings = { command: "rbtr", autoIndex: true };
  let cliAvailable = false;
  let footer: Footer | null = null;
  let healthTimer: ReturnType<typeof setInterval> | null = null;

  /**
   * Try the daemon path, fall back to the CLI callback on
   * transport failure.  Propagates ``RbtrDaemonError`` from the
   * daemon untouched — that is an actionable reply, not a
   * transport problem.
   */
  async function withFallback<T>(fromDaemon: () => Promise<T>, fromCli: () => Promise<T>): Promise<T> {
    if (session.available) {
      try {
        return await fromDaemon();
      } catch (err) {
        if (err instanceof RbtrDaemonError) throw err;
        if (err instanceof DaemonUnavailableError) {
          // Transport failure — fall through to CLI.
        } else {
          throw err;
        }
      }
    }
    if (!resolved || !cliAvailable) {
      throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
    }
    return fromCli();
  }

  function mapDaemonError(err: unknown): ToolReturn {
    if (err instanceof RbtrDaemonError) {
      // A missing ref is fixed by indexing it, so say how.
      const hint = err.code === "index_not_built" ? " Call rbtr_watch with that ref, then retry." : "";
      return {
        content: [{ type: "text", text: `${err.code}: ${err.message}${hint}` }],
        details: { errorCode: err.code, message: err.message },
      };
    }
    throw err;
  }

  async function queryIndexStatus(
    repo: string,
    scope: "workspace" | "all" = "workspace",
  ): Promise<StatusResponse | null> {
    if (session.available) {
      try {
        return await session.send({ kind: "status", repo_path: repo, scope });
      } catch (err) {
        if (err instanceof RbtrDaemonError) return null;
        // transport — fall through to CLI
      }
    }
    if (!resolved) return null;
    try {
      const args = scope === "all" ? ["status", "--scope", "all"] : ["status"];
      return await runRbtrJson<StatusResponse>(pi, resolved, args, { timeout: 5000 });
    } catch {
      return null;
    }
  }

  // ── Session lifecycle ───────────────────────────────────────

  pi.on("session_start", async (_event, ctx) => {
    pi.setActiveTools(startingTools(pi.getActiveTools()));
    settings = loadSettings(ctx.cwd);
    resolved = resolveCommand(settings.command);
    footer = new Footer(ctx);

    // Look for the daemon first — one CLI shell-out here avoids
    // a per-tool-call query.  Failure leaves the session marked
    // unavailable and the tools fall back to CLI exec.
    await session.refresh();

    // Reconcile daemon version with the extension: start if
    // missing, restart if we're newer, yield if older.  See
    // DaemonSession.reconcile for the full rule set.
    if (resolved) {
      const reconcileResult = await session.reconcile(EXTENSION_VERSION, {
        execDaemon: async (sub) => {
          if (!resolved) throw new Error("rbtr command not resolved");
          if (sub === "start") footer?.setSpinner("muted", (frame) => `rbtr: ${frame} starting daemon…`);
          else footer?.setSpinner("muted", (frame) => `rbtr: ${frame} restarting daemon…`);
          const result = await pi.exec(resolved.executable, [...resolved.baseArgs, "daemon", sub], {
            timeout: 15_000,
          });
          if (result.code !== 0) {
            throw new Error(classifyDaemonFailure(result.code, result.stderr ?? "").message);
          }
        },
      });
      notifyReconcile(ctx, reconcileResult);
    }

    if (session.available) {
      startSubscription(ctx);
    }

    healthTimer = setInterval(() => {
      void checkDaemonHealth(ctx);
    }, 30_000);

    const status = await queryIndexStatus(ctx.cwd);
    const decision = decideStartupDecision(resolved !== null, status, settings.autoIndex);

    if (decision.kind === "missing-cli") {
      // Genuinely unresolvable CLI — the only case that disables
      // rbtr for the session.  A transient daemon/lock failure
      // must not land here (it would mislabel a present CLI as
      // missing and skip auto-index).
      cliAvailable = false;
      footer.setStatic("error", "rbtr: not found");
      ctx.ui.notify(
        "rbtr CLI not found. Install with: uv tool install rbtr\n" +
          'Or set command in .pi/rbtr-index.json to "uvx" or "uvx --from <path>"',
        "warning",
      );
      return;
    }

    cliAvailable = true;

    if (decision.kind === "indexed") {
      const top = (status?.indexed_refs ?? [])[0];
      if (top) footer.setStatic("success", footerLabel(top, session.available));
      return;
    }

    if (decision.kind === "transient") {
      ctx.ui.notify("rbtr index temporarily unavailable (database busy); will retry.", "warning");
    }

    // empty or transient: index if enabled (the CLI auto-starts /
    // falls back), else leave a hint in the footer.
    if (decision.index) {
      await triggerIndex(ctx);
    } else {
      const hint =
        decision.kind === "transient" ? "rbtr: index unavailable — retrying" : "rbtr: no index — /rbtr-index";
      footer.setStatic("muted", hint);
    }
  });

  pi.on("session_shutdown", async () => {
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
   */
  async function checkDaemonHealth(ctx: ExtensionContext): Promise<void> {
    const transition = await session.detectTransition();
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
   * them.  Filters to ``notification.repo_path === ctx.cwd`` so we
   * ignore traffic from other repos the daemon might be
   * watching.
   */
  function startSubscription(ctx: ExtensionContext): void {
    let buildStartedAt: number | null = null;

    const elapsedSuffix = (): string => {
      if (buildStartedAt === null) return "";
      return ` · ${formatElapsed(Math.floor((Date.now() - buildStartedAt) / 1000))}`;
    };

    try {
      session.subscribe((notification) => {
        if (notification.repo_path !== ctx.cwd) return;
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

  pi.on("before_agent_start", async (event) => {
    if (!cliAvailable) return;
    return {
      systemPrompt:
        event.systemPrompt +
        "\n\nThis repository has an rbtr code index. Use the rbtr_* tools to find code by meaning, structure or name; use grep for exact strings.",
    };
  });

  /**
   * Submit a build to the daemon (or via CLI fallback).
   *
   * Fire-and-forget: the daemon returns OkResponse immediately
   * and runs the build on its internal queue.  Progress updates
   * arrive via PUB notifications (wired in Phase 8.4).
   */
  async function triggerIndex(ctx: ExtensionContext, ...refs: string[]): Promise<void> {
    const targetRefs = refs.length > 0 ? refs : ["HEAD"];
    footer?.setSpinner("muted", (frame) => `rbtr: ${frame} indexing…`);
    try {
      await withFallback(
        async () => {
          // The extension always builds the 'full' variant (the default).
          // The 'stripped' variant is benchmark-only, driven by rbtr-eval.
          await session.send({ kind: "watch", repo_path: ctx.cwd, refs: targetRefs });
        },
        async () => {
          if (!resolved) throw new Error("rbtr CLI not available");
          const args = ["watch"];
          for (const r of targetRefs) args.push(r);
          await runRbtrJson<WatchResponse>(pi, resolved, args, { timeout: 600_000 });
        },
      );
    } catch (err) {
      footer?.setStatic("error", "rbtr: indexing failed");
      ctx.ui.notify(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }

  // Stop watching the given refs (daemon path, CLI fallback).  The
  // protocol takes at least one ref: unwatching nothing is a mis-shaped
  // call, and finding the stale ones is `triggerRemoveStale`.
  async function triggerUnwatch(ctx: ExtensionContext, refs: [string, ...string[]]): Promise<void> {
    await withFallback(
      async () => {
        await session.send({ kind: "unwatch", repo_path: ctx.cwd, refs });
      },
      async () => {
        if (!resolved) throw new Error("rbtr CLI not available");
        await runRbtr(pi, resolved, ["unwatch", ...refs], { timeout: 60_000 });
      },
    );
  }

  // Drop watched refs that no longer resolve (e.g. deleted branches).
  // Which refs those are is rbtr's to decide, so this asks rather than
  // working it out here; HEAD is never removed.
  async function triggerRemoveStale(ctx: ExtensionContext): Promise<string[]> {
    const removed = await withFallback(
      async () => session.send({ kind: "unwatch_stale", repo_path: ctx.cwd }),
      async () => {
        if (!resolved) throw new Error("rbtr CLI not available");
        return runRbtrJson<UnwatchResponse>(pi, resolved, ["unwatch", "--stale"], {
          timeout: 60_000,
        });
      },
    );
    return Object.values(removed.removed ?? {}).flat();
  }

  async function triggerGc(
    ctx: ExtensionContext,
    opts: { watchedOnly: boolean; dryRun: boolean },
  ): Promise<GcResponse> {
    const mode: GcMode = opts.watchedOnly ? "watched_only" : "watched";
    return withFallback(
      async () => session.send({ kind: "gc", repo_path: ctx.cwd, mode, refs: [], dry_run: opts.dryRun }),
      async () => {
        if (!resolved) throw new Error("rbtr CLI not available");
        const args = ["gc"];
        if (opts.watchedOnly) args.push("--keep", "watched-only");
        if (opts.dryRun) args.push("--dry-run");
        return runRbtrJson<GcResponse>(pi, resolved, args, { timeout: 120_000 });
      },
    );
  }

  // ── Commands ────────────────────────────────────────────────

  pi.registerCommand("rbtr-status", {
    description: "Show rbtr index status",
    handler: async (_args, ctx) => {
      if (!cliAvailable) {
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
      if (!cliAvailable) {
        ctx.ui.notify("rbtr CLI not available", "error");
        return;
      }
      const refs = commandRefs(args);
      await triggerIndex(ctx, ...refs);
      ctx.ui.notify(`Indexing ${refs.length > 0 ? refs.join(", ") : "HEAD"}. Progress in the footer.`, "info");
    },
  });

  pi.registerCommand("rbtr-settings", {
    description: "View and toggle rbtr index settings",
    handler: async (_args, ctx) => {
      if (!ctx.hasUI) {
        ctx.ui.notify("/rbtr-settings requires interactive mode", "error");
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
              `${theme.fg("muted", "CLI available:")} ${cliAvailable ? theme.fg("success", "yes") : theme.fg("error", "no")}`,
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
    renderCall: (args, theme) => renderSearchCall(args, theme),
    renderResult: (result, options, theme) => renderSearchResult(result, options, theme),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      if (!params.query) throw new Error("Missing required parameter `query`. Example: {query: 'retry logic'}");
      try {
        return await withFallback<ToolReturn>(
          async () => {
            const resp = await session.send({
              kind: "search",
              repo_path: ctx.cwd,
              query: params.query,
              ...(params.ref !== undefined ? { ref: params.ref } : {}),
              ...(params.limit !== undefined ? { limit: params.limit } : {}),
              ...(params.keywords !== undefined ? { keywords: params.keywords } : {}),
              ...(params.variants !== undefined ? { variants: params.variants } : {}),
              ...(params.scope !== undefined ? { scope: params.scope } : {}),
            });
            if (resp.results.length === 0) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No results found.${echoArgs(params, ["query", "ref", "keywords", "variants", "scope"])}`,
                  },
                ],
                details: { fromDaemon: true, response: resp },
              };
            }
            return toolResultFromDaemon(resp);
          },
          async () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const args = ["search", params.query];
            if (params.ref !== undefined) args.push("--ref", params.ref);
            if (params.limit !== undefined) args.push("--limit", String(params.limit));
            if (params.scope !== undefined) args.push("--scope", params.scope);
            const result = await runRbtr(pi, resolved, args, { signal, timeout: READ_CLI_TIMEOUT_MS });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No results found.${echoArgs(params, ["query", "ref", "keywords", "variants", "scope"])}`,
                  },
                ],
                details: { fromCli: true, results: [] },
              };
            }
            return toolResultFromCli(text, { query: params.query, limit: params.limit });
          },
        );
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_read_symbol",
    label: "rbtr read-symbol",
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
    renderCall: (args, theme) => renderReadSymbolCall(args, theme),
    renderResult: (result, options, theme) => renderReadSymbolResult(result, options, theme),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      if (!params.symbol) throw new Error("Missing required parameter `symbol`. Example: {symbol: 'MyClass.method'}");
      try {
        return await withFallback<ToolReturn>(
          async () => {
            const resp = await session.send({
              kind: "read_symbol",
              repo_path: ctx.cwd,
              symbol: params.symbol,
              ...(params.ref !== undefined ? { ref: params.ref } : {}),
              ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
            });
            if (resp.chunks.length === 0) {
              return {
                content: [
                  {
                    type: "text",
                    text: `Symbol not found: ${params.symbol}${echoArgs(params, ["ref", "file_paths"])}`,
                  },
                ],
                details: { fromDaemon: true, response: resp, symbol: params.symbol },
              };
            }
            return toolResultFromDaemon(resp);
          },
          async () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const readArgs = ["read-symbol", params.symbol];
            if (params.ref !== undefined) readArgs.push("--ref", params.ref);
            for (const fp of params.file_paths ?? []) readArgs.push("--file-path", fp);
            const result = await runRbtr(pi, resolved, readArgs, { signal, timeout: READ_CLI_TIMEOUT_MS });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [
                  {
                    type: "text",
                    text: `Symbol not found: ${params.symbol}${echoArgs(params, ["ref", "file_paths"])}`,
                  },
                ],
                details: { fromCli: true, symbol: params.symbol, found: false },
              };
            }
            return toolResultFromCli(text, { symbol: params.symbol, found: true });
          },
        );
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_find_refs",
    label: "rbtr find-refs",
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
    renderCall: (args, theme) => renderFindRefsCall(args, theme),
    renderResult: (result, options, theme) => renderFindRefsResult(result, options, theme),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      if (!params.symbol) throw new Error("Missing required parameter `symbol`. Example: {symbol: 'MyClass.method'}");
      try {
        return await withFallback<ToolReturn>(
          async () => {
            const resp = await session.send({
              kind: "find_refs",
              repo_path: ctx.cwd,
              symbol: params.symbol,
              ...(params.ref !== undefined ? { ref: params.ref } : {}),
              ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
            });
            if (resp.refs.length === 0) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No references found for: ${params.symbol}${echoArgs(params, ["ref", "file_paths"])}`,
                  },
                ],
                details: { fromDaemon: true, response: resp },
              };
            }
            return toolResultFromDaemon(resp);
          },
          async () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const findArgs = ["find-refs", params.symbol];
            if (params.ref !== undefined) findArgs.push("--ref", params.ref);
            for (const fp of params.file_paths ?? []) findArgs.push("--file-path", fp);
            const result = await runRbtr(pi, resolved, findArgs, { signal, timeout: READ_CLI_TIMEOUT_MS });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No references found for: ${params.symbol}${echoArgs(params, ["ref", "file_paths"])}`,
                  },
                ],
                details: { fromCli: true, symbol: params.symbol, found: false },
              };
            }
            return toolResultFromCli(text, { symbol: params.symbol, found: true });
          },
        );
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_changed_symbols",
    label: "rbtr changed-symbols",
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
    renderCall: (args, theme) => renderChangedSymbolsCall(args, theme),
    renderResult: (result, options, theme) => renderChangedSymbolsResult(result, options, theme),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      if (!params.base || !params.head) {
        throw new Error(
          "Missing required parameters `base` and `head`. Example: {base: 'main', head: 'feature-branch'}",
        );
      }
      try {
        return await withFallback<ToolReturn>(
          async () => {
            const resp = await session.send({
              kind: "changed_symbols",
              repo_path: ctx.cwd,
              base: params.base,
              head: params.head,
              ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
            });
            if (resp.changes.length === 0) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No changed symbols between ${params.base} and ${params.head}${echoArgs(params, ["file_paths"])}`,
                  },
                ],
                details: { fromDaemon: true, response: resp },
              };
            }
            return toolResultFromDaemon(resp);
          },
          async () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const changedArgs = ["changed-symbols", params.base, params.head];
            for (const fp of params.file_paths ?? []) changedArgs.push("--file-path", fp);
            const result = await runRbtr(pi, resolved, changedArgs, {
              signal,
              timeout: READ_CLI_TIMEOUT_MS,
            });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No changed symbols between ${params.base} and ${params.head}${echoArgs(params, ["file_paths"])}`,
                  },
                ],
                details: { fromCli: true, base: params.base, head: params.head, found: false },
              };
            }
            return toolResultFromCli(text, { base: params.base, head: params.head, found: true });
          },
        );
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_list_symbols",
    label: "rbtr list-symbols",
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
    renderCall: (args, theme) => renderListSymbolsCall(args, theme),
    renderResult: (result, options, theme) => renderListSymbolsResult(result, options, theme),

    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      if (!params.file)
        throw new Error("Missing required parameter `file`. Example: {file: 'src/rbtr/index/search.py'}");
      try {
        return await withFallback<ToolReturn>(
          async () => {
            const resp = await session.send({
              kind: "list_symbols",
              repo_path: ctx.cwd,
              file_path: params.file,
              ...(params.ref !== undefined ? { ref: params.ref } : {}),
            });
            if (resp.chunks.length === 0) {
              return {
                content: [{ type: "text", text: `No symbols found in: ${params.file}` }],
                details: { fromDaemon: true, response: resp },
              };
            }
            return toolResultFromDaemon(resp);
          },
          async () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const listArgs = ["list-symbols", params.file];
            if (params.ref !== undefined) listArgs.push("--ref", params.ref);
            const result = await runRbtr(pi, resolved, listArgs, {
              signal,
              timeout: READ_CLI_TIMEOUT_MS,
            });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [{ type: "text", text: `No symbols found in: ${params.file}` }],
                details: { fromCli: true, file: params.file, found: false },
              };
            }
            return toolResultFromCli(text, { file: params.file, found: true });
          },
        );
      } catch (err) {
        return mapDaemonError(err);
      }
    },
  });

  pi.registerTool({
    name: "rbtr_watch",
    label: "rbtr watch",
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
    renderCall: (args, theme) => renderIndexCall(args, theme),
    renderResult: (result, options, theme) => renderIndexResult(result, options, theme),

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!cliAvailable) {
        throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
      }
      // Decode refs in case the provider delivered the array as a
      // JSON-encoded string, so the up-to-date check and message below
      // operate on the real refs (the daemon decodes its copy too).
      const decodedRefs = decodeStringList(params.refs);
      const refs: [string, ...string[]] = decodedRefs.length > 0 ? [decodedRefs[0], ...decodedRefs.slice(1)] : ["HEAD"];

      if (params.remove_stale) {
        const pruned = await triggerRemoveStale(ctx);
        return {
          content: [
            {
              type: "text",
              text: pruned.length > 0 ? `Stopped watching stale refs: ${pruned.join(", ")}.` : "No stale watched refs.",
            },
          ],
          details: { status: "remove_stale", pruned },
        };
      }
      if (params.remove) {
        if (refs.includes("HEAD")) {
          return {
            content: [{ type: "text", text: "HEAD cannot be removed from the watch set." }],
            details: { status: "rejected", refs },
          };
        }
        await triggerUnwatch(ctx, refs);
        return {
          content: [{ type: "text", text: `Stopped watching: ${refs.join(", ")}.` }],
          details: { status: "removed", refs },
        };
      }

      // Check current state first so we can give the LLM an
      // actionable answer instead of blindly queueing a build.
      // The daemon already dedupes duplicate submits (phase 9.2);
      // this just turns that silent deduplication into a clear
      // signal.
      const status = await queryIndexStatus(ctx.cwd);

      if (status?.active_build && status.active_build.repo_path === ctx.cwd) {
        const j = status.active_build;
        return {
          content: [
            {
              type: "text",
              text:
                `A build is already in progress for this repository at ${shortSha(j.ref)} ` +
                `(${j.phase} ${formatJobCounts(j)}). No new build was queued. ` +
                `Use rbtr_status to check progress.`,
            },
          ],
          details: { status: "in_progress", refs, activeJob: j },
        };
      }

      // If asking for HEAD and HEAD is already indexed, say so
      // rather than pretending to queue a redundant build.
      if (refs.length === 1 && refs[0] === "HEAD" && status?.indexed_refs && status.indexed_refs.length > 0) {
        const headRef = status.indexed_refs.find((r) => (r.names ?? []).includes("HEAD"));
        if (headRef) {
          return {
            content: [
              {
                type: "text",
                text: `Index is up to date for HEAD (${shortSha(headRef.sha)}). No action taken.`,
              },
            ],
            details: { status: "up_to_date", refs, head: headRef.sha },
          };
        }
      }

      await triggerIndex(ctx, ...refs);
      return {
        content: [
          {
            type: "text",
            text: `Indexing queued for refs ${refs.join(", ")}. Progress is shown in the footer. Use rbtr_status to check when complete.`,
          },
        ],
        details: { status: "started", refs },
      };
    },
  });

  pi.registerTool({
    name: LOADER,
    label: "rbtr index tools",
    description:
      "Load the tools that inspect and clean up the rbtr index: rbtr_status (what is indexed, and whether a build is still running) and rbtr_gc (reclaim disk space; destructive, only on the user's request).",
    promptSnippet: "Load rbtr_status and rbtr_gc, to inspect the index or reclaim its disk space",
    promptGuidelines: [
      "Call rbtr_index_tools when you need to check whether indexing has finished, or the user asks to reclaim index space.",
    ],
    parameters: Type.Object({}),

    async execute() {
      pi.setActiveTools(withOnRequest(pi.getActiveTools()));
      return {
        content: [
          {
            type: "text",
            text: "Loaded rbtr_status (what is indexed, and whether a build is still running) and rbtr_gc (reclaim index space; destructive, only on the user's request).",
          },
        ],
        details: { loaded: [...ON_REQUEST] },
      };
    },
  });

  pi.registerTool({
    name: "rbtr_status",
    label: "rbtr status",
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
    renderCall: (args, theme) => renderStatusCall(args, theme),
    renderResult: (result, options, theme) => renderStatusResult(result, options, theme),

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!cliAvailable) {
        throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
      }
      const status = await queryIndexStatus(ctx.cwd, params.scope ?? "workspace");
      if (!status) {
        throw new Error("Failed to check index status");
      }
      return {
        content: [{ type: "text", text: renderStatusText(status) }],
        details: { fromDaemon: true, response: status },
      };
    },
  });

  pi.registerTool({
    name: "rbtr_gc",
    label: "rbtr gc",
    description:
      "⚠ DESTRUCTIVE, IRREVERSIBLE. Permanently deletes indexed data (commits, chunks, embeddings) from the rbtr index; there is no undo. Only for when the user has explicitly asked to reclaim index space. Runs as a dry-run preview by default: show the user what it would drop, and pass dry_run=false only after they confirm. By default keeps HEAD, all local branches and tags, and the watch set, dropping only unreferenced commits; watched_only also drops unwatched branches and tags. The daemon never deletes anything unless this is called.",
    promptSnippet: "Delete unreferenced index data to reclaim disk space (destructive; only on the user's request)",
    promptGuidelines: [
      "Use rbtr_gc only when the user explicitly asks to reclaim index space, and apply it only after they confirm the dry-run result.",
    ],
    parameters: Type.Object({
      watched_only: Type.Optional(
        Type.Boolean({ description: "Keep only HEAD and watched refs (drop unwatched branches/tags)." }),
      ),
      dry_run: Type.Optional(
        Type.Boolean({
          description:
            "Preview without deleting. Defaults to true (safe); set false ONLY after explicit user confirmation to actually delete.",
        }),
      ),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      if (!cliAvailable) {
        throw new Error("rbtr CLI not available. Install with: uv tool install rbtr");
      }
      // Dry-run by default: a careless call previews; deleting needs a
      // deliberate dry_run=false after the user has confirmed.
      const res = await triggerGc(ctx, {
        watchedOnly: params.watched_only ?? false,
        dryRun: params.dry_run ?? true,
      });
      const text = res.dry_run
        ? `Dry run: would drop ${res.snapshots_dropped} snapshot(s) and free ${res.chunks_freed} chunk(s) across ${res.repos_collected} repo(s). Nothing was deleted — confirm with the user, then call again with dry_run=false to apply.`
        : `Dropped ${res.snapshots_dropped} snapshot(s); freed ${res.chunks_freed} chunk(s) across ${res.repos_collected} repo(s).`;
      return {
        content: [{ type: "text", text }],
        details: { fromDaemon: true, response: res },
      };
    },
  });
}
