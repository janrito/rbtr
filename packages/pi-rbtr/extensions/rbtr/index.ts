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
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  getSettingsListTheme,
  truncateHead,
} from "@mariozechner/pi-coding-agent";
import { Container, type SettingItem, SettingsList } from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";

import { classifyDaemonFailure, decideStartupDecision } from "./classify.js";
import { RbtrDaemonError } from "./daemon-client.js";
import { DaemonSession, DaemonUnavailableError, type ReconcileResult } from "./daemon-session.js";
import { type ResolvedCommand, resolveCommand, runRbtr, runRbtrJson } from "./exec.js";
import { Footer } from "./footer.js";
import type { BuildIndexResponse, GcMode, GcResponse, Response, StatusResponse } from "./generated/protocol.js";

const require = createRequire(import.meta.url);
const { version: EXTENSION_VERSION } = require("../../package.json") as { version: string };

import { decodeStringList, echoArgs } from "./args.js";
import {
  formatWatched,
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

/**
 * Render a `StatusResponse` as multi-line text for the LLM.
 *
 * Output is derived solely from the response model — no
 * external state.  Mirrors the Python CLI shape so the model
 * sees the same information regardless of transport.
 */
function renderStatusText(status: StatusResponse): string {
  const lines: string[] = [];
  const indexed = status.indexed_refs ?? [];
  if (indexed.length === 0) {
    lines.push("No index found at the configured path.");
  } else {
    const total = indexed[0].total;
    lines.push(`Index: ${humanCount(total)} symbols (${status.db_path})`);
    lines.push("Refs:");
    for (const ref of indexed) {
      const label =
        (ref.names ?? []).length > 0
          ? `${ref.sha.slice(0, 12)} (${(ref.names ?? []).join(", ")})`
          : ref.sha.slice(0, 12);
      const embedPart =
        ref.embedded >= ref.total
          ? `${humanCount(ref.embedded)} embedded`
          : ref.embedded > 0
            ? `${humanCount(ref.embedded)} embedded (${Math.round((100 * ref.embedded) / ref.total)}%)`
            : "not embedded";
      lines.push(`  ${label} — ${humanCount(ref.total)} indexed, ${embedPart}`);
    }
  }
  lines.push(...formatWatched(status.watched ?? []));
  const job = status.active_build;
  if (job) {
    const pct = job.total > 0 ? ` (${Math.round((100 * job.current) / job.total)}%)` : "";
    const elapsed = formatElapsed(job.elapsed_seconds);
    lines.push(`Building: ${job.ref.slice(0, 12)} — ${job.phase} ${job.current}/${job.total}${pct} — ${elapsed}`);
  }
  const ej = status.active_embed;
  if (ej) {
    const pct = ej.total > 0 ? ` (${Math.round((100 * ej.current) / ej.total)}%)` : "";
    const elapsed = formatElapsed(ej.elapsed_seconds);
    lines.push(`Embedding: ${ej.ref.slice(0, 12)} — ${ej.current}/${ej.total}${pct} — ${elapsed}`);
  }
  if (!job && !ej && indexed.length > 0) {
    lines.push("No active build.");
  }
  return lines.join("\n");
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m${String(s).padStart(2, "0")}s`;
}

function humanCount(n: number): string {
  if (n < 1000) return String(n);
  return `${(n / 1000).toFixed(1)}k`;
}

/**
 * Format the footer label for an indexed repo.
 *
 * Glyph reflects index + embedding completeness:
 *   ● — fully indexed and fully embedded
 *   ○ — indexed but not (fully) embedded
 *
 * Suffixes after ` · ` for additional state:
 *   rbtr: ● 3.5k symbols
 *   rbtr: ○ 3.5k symbols · not embedded
 *   rbtr: ○ 3.5k symbols · 42% embedded
 *   rbtr: ● 3.5k symbols · no daemon
 *   rbtr: ○ 3.5k symbols · no daemon · not embedded
 */
function footerLabel(total: number, embedded: number, daemon: boolean): string {
  const glyph = embedded >= total ? "●" : "○";
  const parts = [`rbtr: ${glyph} ${humanCount(total)} symbols`];
  if (!daemon) parts.push("no daemon");
  if (embedded < total) {
    parts.push(embedded > 0 ? `${Math.round((100 * embedded) / total)}% embedded` : "not embedded");
  }
  return parts.join(" · ");
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
      return {
        content: [{ type: "text", text: `${err.code}: ${err.message}` }],
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
      if (top) footer.setStatic("success", footerLabel(top.total, top.embedded, session.available));
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
          footer?.setStatic("success", footerLabel(indexed[0].total, indexed[0].embedded, false));
        } else {
          footer?.setStatic("muted", "rbtr: no index · no daemon");
        }
        break;
      }
      case "returned": {
        const status = await queryIndexStatus(ctx.cwd);
        const indexed = status?.indexed_refs ?? [];
        if (indexed.length > 0) {
          footer?.setStatic("success", footerLabel(indexed[0].total, indexed[0].embedded, true));
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
      const s = Math.floor((Date.now() - buildStartedAt) / 1000);
      if (s < 60) return ` · ${s}s`;
      const m = Math.floor(s / 60);
      const rem = s % 60;
      return ` · ${m}m${String(rem).padStart(2, "0")}s`;
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
          case "embed_complete":
            buildStartedAt = null;
            footer.setStatic("success", footerLabel(notification.chunks, notification.embedded, true));
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
        "\n\nThis repository has an rbtr structural code index available. " +
        "Prefer the rbtr_* tools over grep + read for code navigation:\n" +
        '- Concept-shaped questions ("how is X handled", "where does Y happen", "find the retry logic") → rbtr_search. ' +
        "It understands meaning, not just substrings.\n" +
        "- Known symbol by name → rbtr_read_symbol. No file path needed.\n" +
        "- File outline before loading → rbtr_list_symbols.\n" +
        "- Callers / tests / doc mentions of a symbol → rbtr_find_refs.\n" +
        "- Structural diff between refs (for PR review) → rbtr_changed_symbols.\n" +
        "Keep grep for literal strings (error messages, config keys, regexes) and use `read` for full-file reads; " +
        "rbtr_* tools are for structure. When a concept-question arises, reach for rbtr_search first.",
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
          await session.send({ kind: "index", repo_path: ctx.cwd, refs: targetRefs });
        },
        async () => {
          if (!resolved) throw new Error("rbtr CLI not available");
          const args = ["index"];
          for (const r of targetRefs) args.push(r);
          await runRbtrJson<BuildIndexResponse>(pi, resolved, args, { timeout: 600_000 });
        },
      );
    } catch (err) {
      footer?.setStatic("error", "rbtr: indexing failed");
      ctx.ui.notify(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }

  // Stop watching the given refs (daemon path, CLI fallback).
  async function triggerUnwatch(ctx: ExtensionContext, refs: string[]): Promise<void> {
    await withFallback(
      async () => {
        await session.send({ kind: "index", repo_path: ctx.cwd, refs, remove: true });
      },
      async () => {
        if (!resolved) throw new Error("rbtr CLI not available");
        await runRbtr(pi, resolved, ["index", "--remove", ...refs], { timeout: 60_000 });
      },
    );
  }

  // Prune watched refs that no longer resolve (e.g. deleted branches).
  // Reuses status (unresolvable → sha === null) then unwatches them;
  // HEAD always resolves, so it is never pruned.
  async function triggerRemoveStale(ctx: ExtensionContext): Promise<string[]> {
    const status = await queryIndexStatus(ctx.cwd);
    const stale = (status?.watched ?? []).filter((w) => w.sha == null && w.ref !== "HEAD").map((w) => w.ref);
    if (stale.length > 0) await triggerUnwatch(ctx, stale);
    return stale;
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
        if (opts.watchedOnly) args.push("--watched-only");
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
    description: "Index the repository (or rebuild the index)",
    handler: async (_args, ctx) => {
      if (!cliAvailable) {
        ctx.ui.notify("rbtr CLI not available", "error");
        return;
      }
      await triggerIndex(ctx);
      ctx.ui.notify("Indexing started. Progress in the footer.", "info");
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
    name: "rbtr_index",
    label: "rbtr index",
    description:
      "Manage the rbtr watch set: keep the given refs (branch, tag, or SHA) indexed so the other rbtr_* tools work. With no refs it watches HEAD. `remove` stops watching refs; `remove_stale` drops refs that no longer resolve. Safe to call repeatedly.",
    promptSnippet: "Watch refs for the rbtr code index (or --remove to stop)",
    promptGuidelines: [
      "Call rbtr_index when the user asks to (re)index or watch a specific ref, or when another rbtr_* tool returns a 'not indexed' error. The daemon keeps watched refs (HEAD by default) indexed automatically — you rarely need this for HEAD.",
      "Each positional ref is an independent watch target the daemon keeps current; a branch tracks its tip, a bare SHA settles after one build. HEAD is always watched and cannot be removed.",
      "When you begin substantive work on a branch, watch its base too (the default branch it forked from), not just HEAD — so you can later review the branch with rbtr_changed_symbols without a cold index.",
      "Use `remove` to stop watching refs, or `remove_stale` to drop refs whose branch was deleted.",
      "When a watched branch has been merged or no longer resolves (e.g. after a merge), suggest the user stop watching it — `remove_stale` for deleted branches, or `remove` for a specific ref. This only trims the watch set; it doesn't delete index data.",
      "This is fire-and-forget: the tool returns immediately. Use rbtr_status to see progress and the current watch set.",
    ],
    parameters: Type.Object({
      refs: Type.Optional(
        Type.Array(Type.String(), {
          description: "Refs to watch and index (default: ['HEAD']). Each ref is an independent watch target.",
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
      const refs = decodedRefs.length > 0 ? decodedRefs : ["HEAD"];

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
        const pct = j.total > 0 ? ` (${Math.round((100 * j.current) / j.total)}%)` : "";
        return {
          content: [
            {
              type: "text",
              text:
                `A build is already in progress for this repository at ${j.ref.slice(0, 12)} ` +
                `(${j.phase} ${j.current}/${j.total}${pct}). No new build was queued. ` +
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
                text: `Index is up to date for HEAD (${headRef.sha.slice(0, 12)}). No action taken.`,
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
    name: "rbtr_status",
    label: "rbtr status",
    description:
      "Report the state of the rbtr index for this repository: total symbols, which commits are indexed, any build currently running (phase + progress + elapsed), and pending queue entries.",
    promptSnippet: "Inspect rbtr code-index state: totals, indexed commits, active build, queue",
    promptGuidelines: [
      "Call rbtr_status when you need to know whether the index is ready, whether a build you just queued is done, or which refs are indexed (e.g. before rbtr_changed_symbols).",
      "If the reply shows 'Building: ...' with phase/progress, a build is live — don't re-queue it; either wait and re-check, or answer the user without the index for now.",
      "If 'Indexed commits: none' and no active job, the user has never completed an index; tell them to run rbtr_index (or it will happen automatically at session start unless disabled).",
      "Leave `scope` as the default 'workspace' (the current repo) unless you specifically need to see other indexed repos. Set `scope: 'all'` only to confirm which other projects are indexed before a deliberate cross-repo rbtr_search — i.e. when the user's task spans sibling checkouts, a split monorepo, or an explicit cross-project question.",
    ],
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
      "⚠ DESTRUCTIVE, IRREVERSIBLE. Permanently deletes indexed data (commits, chunks, embeddings) from the rbtr index — there is no undo. Runs as a dry-run preview by default; only pass dry_run=false when the user has EXPLICITLY asked to delete index data. By default keeps HEAD, all local branches/tags, and the watch set (drops only unreferenced commits); watched_only also drops unwatched branches/tags.",
    promptSnippet: "Garbage-collect the rbtr index (⚠ destructive; reclaim storage)",
    promptGuidelines: [
      "⚠ Destructive and irreversible. Only call rbtr_gc when the user has EXPLICITLY asked to reclaim index space or drop refs — never speculatively, never to 'tidy up' on your own initiative, never as a side-effect of another task.",
      "It runs as a dry-run preview by default. Show the user what it would drop and get their explicit confirmation; pass dry_run=false ONLY after they confirm they want the deletion applied.",
      "The default mode never drops anything reachable from a branch or the watch set; watched_only=true additionally drops unwatched branches/tags. The daemon never GCs automatically — nothing is deleted unless you call this.",
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
        ? `Dry run: would drop ${res.commits_dropped} commit(s); ${res.chunks_dropped} chunk(s) freed, ${res.chunks_kept_shared} chunk(s) kept. Nothing was deleted — confirm with the user, then call again with dry_run=false to apply.`
        : `Dropped ${res.commits_dropped} commit(s); ${res.chunks_dropped} chunk(s) freed, ${res.chunks_kept_shared} chunk(s) kept.`;
      return {
        content: [{ type: "text", text }],
        details: { fromDaemon: true, response: res },
      };
    },
  });

  pi.registerTool({
    name: "rbtr_search",
    label: "rbtr search",
    description:
      "Semantic + symbol-aware search over the repository's code index. Takes a `query` string and optional `keywords`/`variants` for query expansion. Ranks functions, classes, methods, and other structural chunks by relevance (embeddings + full-text fused). Returns scored hits with file path, line, kind, name, and full source content.",
    promptSnippet: "Semantic code search by intent: 'how retries are handled', 'where embeddings happen'",
    promptGuidelines: [
      "Prefer rbtr_search over grep for concept-shaped questions: 'how does X work', 'where is Y handled', 'find the code that does Z'. It understands meaning, not just substrings — 'retry logic' finds 'backoff' and 'reconnect attempts', grep would miss them.",
      "Use grep when the user gives you a literal string or identifier that must match exactly (an error message, a regex, a configuration key, an import path).",
      "Use rbtr_read_symbol (not rbtr_search) when you already know a symbol name and want its full source.",
      "Chain: rbtr_search finds candidates → pass the `name` field from a hit as the `symbol` parameter to rbtr_read_symbol for the full source → rbtr_find_refs for callers / tests.",
      "Scores are meaningful relative to the top result. A big drop-off after the first few hits means the tail is probably noise.",
      "For concept queries (natural language like 'how does idle unload work'): provide `keywords` (3-5 synonym identifiers or alternative terms, e.g. ['timeout', 'evict', 'unload_model']) and `variants` (1-2 rephrases using different terminology, e.g. ['when does the model get released from memory']).",
      "For identifier queries (symbol names like `fuse_scores`): optionally provide `keywords` only (alternative names the symbol might have, e.g. ['merge_scores', 'combine_results']). Omit `variants`.",
      "For code fragments (e.g. `for item in self._cache`): omit both `keywords` and `variants`.",
      "Leave `scope` as the default 'workspace' for almost every search: the user is working in this repo and wants answers from it. 'workspace' searches only the current project.",
      "Only set `scope: 'all'` (search every indexed repo, merged into one ranked list with each hit labelled by repo) when the task genuinely spans repos: the user is working across sibling checkouts of one system (e.g. a frontend and its backend), a monorepo that was split into separate checkouts, or has explicitly asked how this project integrates with or depends on another indexed one. Cross-repo results pull in code from unrelated projects and dilute relevance, so do not reach for 'all' just because a 'workspace' search came back thin — refine the query first.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
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
                    text: `No results found.${echoArgs(params, ["query", "keywords", "variants", "scope"])}`,
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
            if (params.limit !== undefined) args.push("--limit", String(params.limit));
            if (params.scope !== undefined) args.push("--scope", params.scope);
            const result = await runRbtr(pi, resolved, args, { signal, timeout: 30_000 });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No results found.${echoArgs(params, ["query", "keywords", "variants", "scope"])}`,
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
      "Fetch the full source of a named symbol (function, class, method, constant) from the code index. Takes a `symbol` string — just the name, no file path needed (e.g. 'fuse_scores', 'HttpClient.retry').",
    promptSnippet: "Read a symbol's source by name: 'fuse_scores', 'HttpClient.retry'",
    promptGuidelines: [
      "Use rbtr_read_symbol when you have a symbol name and want its source. Much more precise than grep + read — no guessing at line numbers, no page-through.",
      "Symbol names come as: bare name (`fuse_scores`), class-qualified (`HttpClient.retry`), or module-qualified (`rbtr.index.search.fuse_scores`). The index stores whatever the language's tree-sitter parser emits; try the plain name first, then qualify if there are collisions.",
      "Typical chain: rbtr_search → pick a hit → pass its `name` field as the `symbol` parameter to rbtr_read_symbol for the full body. Then rbtr_find_refs on the same name to see callers.",
      "If the tool returns 'Symbol not found', the symbol either doesn't exist at the indexed ref or lives in a file type the parser doesn't cover; fall back to grep + read.",
      "Pass file_paths to disambiguate a name that exists in several files — only symbols defined in the listed files are returned.",
    ],
    parameters: Type.Object({
      symbol: Type.String({
        description:
          "Symbol name as stored in the index. Examples: 'fuse_scores', 'HttpClient.retry', 'rbtr.index.search.fuse_scores'.",
      }),
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
              ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
            });
            if (resp.chunks.length === 0) {
              return {
                content: [
                  { type: "text", text: `Symbol not found: ${params.symbol}${echoArgs(params, ["file_paths"])}` },
                ],
                details: { fromDaemon: true, response: resp, symbol: params.symbol },
              };
            }
            return toolResultFromDaemon(resp);
          },
          async () => {
            if (!resolved) throw new Error("rbtr CLI not available");
            const readArgs = ["read-symbol", params.symbol];
            for (const fp of params.file_paths ?? []) readArgs.push("--file-path", fp);
            const result = await runRbtr(pi, resolved, readArgs, { signal, timeout: 30_000 });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [
                  { type: "text", text: `Symbol not found: ${params.symbol}${echoArgs(params, ["file_paths"])}` },
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
      "Walk the dependency graph to find every place a symbol is imported, called from tests, or mentioned in docs. Takes a `symbol` string (same format as rbtr_read_symbol). Returns structural edges (source → target, kind=imports|tests|docs), not text matches.",
    promptSnippet: "Find who imports / tests / documents a symbol via the index's dependency graph",
    promptGuidelines: [
      "Use rbtr_find_refs for impact analysis: 'what would break if I change X', 'which tests cover X', 'is X documented anywhere'.",
      "This is structural, not textual: edge kinds ('imports', 'tests', 'docs') give you intent, not raw string occurrences. Prefer this to grep when asking a graph-shaped question.",
      "Use grep when you need every raw occurrence of an identifier including inside strings / comments / unsupported file types.",
      "Chain after rbtr_search or rbtr_read_symbol: you've identified the symbol, now find who depends on it.",
      "Pass file_paths to disambiguate a name that exists in several files — references are resolved only against symbols defined in the listed files.",
    ],
    parameters: Type.Object({
      symbol: Type.String({
        description: "Symbol name (same format as rbtr_read_symbol: bare / class-qualified / module-qualified).",
      }),
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
              ...(params.file_paths !== undefined ? { file_paths: params.file_paths } : {}),
            });
            if (resp.refs.length === 0) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No references found for: ${params.symbol}${echoArgs(params, ["file_paths"])}`,
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
            for (const fp of params.file_paths ?? []) findArgs.push("--file-path", fp);
            const result = await runRbtr(pi, resolved, findArgs, { signal, timeout: 30_000 });
            const text = result.stdout.trim();
            if (!text) {
              return {
                content: [
                  {
                    type: "text",
                    text: `No references found for: ${params.symbol}${echoArgs(params, ["file_paths"])}`,
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
      "Structural diff between two git refs: which functions, classes, or methods were added, modified, or removed. Takes `base` and `head` ref strings (branch name, tag, or SHA). Higher signal than a line diff for reviewing or understanding a branch.",
    promptSnippet: "Symbol-level diff between two refs, for PR review / branch understanding",
    promptGuidelines: [
      "Use rbtr_changed_symbols for code review and branch-understanding questions ('what did this PR change', 'summarise the work on this branch'). It gives you a short list of changed symbols instead of a huge patch to read.",
      "Both refs must be indexed. When you set out to review or understand a branch, watch both refs up front — call rbtr_index with the working ref and its base (usually the default branch it forked from; check the repo, don't assume a name). HEAD is watched automatically. If the tool still errors 'not indexed', index the missing ref and retry.",
      "Use git diff when you need the exact line-level changes or when you need to see non-code changes (config, data files). Use rbtr_changed_symbols when the question is about code structure.",
      "Chain: rbtr_changed_symbols → rbtr_read_symbol on the most interesting entries for the new body → rbtr_find_refs to see callers that might need updating.",
      "Pass file_paths to scope the diff to specific files — only changes in the listed files are reported.",
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
              timeout: 30_000,
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
      "Structural table of contents for a single file. Takes a `file` path relative to repo root (e.g. 'src/rbtr/index/search.py'). Lists every function, class, and method with its line range. Cheap overview of a file's shape without loading its full source.",
    promptSnippet: "File outline: every function/class/method with line ranges (no source)",
    promptGuidelines: [
      "Use rbtr_list_symbols before reading a large file. A few KB of structure tells you whether the content you need is actually in this file before you pay to load it.",
      "Good starting point when a user asks 'what's in file X' or 'walk me through X.py' — list symbols, then rbtr_read_symbol on the ones that matter.",
      "Don't use it for small files (<200 lines); just read them.",
    ],
    parameters: Type.Object({
      file: Type.String({ description: "File path relative to the repo root (e.g. 'src/rbtr/index/search.py')." }),
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
            const result = await runRbtr(pi, resolved, ["list-symbols", params.file], {
              signal,
              timeout: 30_000,
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
}
