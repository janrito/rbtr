/**
 * Session-level state for the daemon client.
 *
 * The extension reuses one `DaemonSession` across tool calls
 * within a pi session.  It caches the RPC endpoint (set at
 * `session_start`) and the PUB endpoint (so Phase 8.4 can
 * attach a subscriber), invalidating on transport errors.
 */

import { Subscriber as ZmqSubscriber } from "zeromq";

import { type DaemonStatus, queryDaemonStatus, send } from "./daemon-client.js";
import type { ResolvedCommand } from "./exec.js";
import type { Notification, Request, Response } from "./generated/protocol.js";

/**
 * A transport-level failure talking to the daemon — socket
 * refused, send/recv timeout, JSON parse.  Distinct from
 * ``RbtrReplyError`` which means the daemon replied with a
 * typed error code.
 */
export class DaemonUnavailableError extends Error {
  constructor(cause: unknown) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    super(`daemon unavailable: ${detail}`);
    this.name = "DaemonUnavailableError";
    this.cause = cause;
  }
}

/** Narrow ``Response`` to the variant for a request of *kind* K. */
type ResponseFor<K extends string> = Extract<Response, { kind: K }>;

/**
 * Outcome of {@link DaemonSession.reconcile}.  Purely informational —
 * callers use it to decide whether to log or notify the user.
 */
export type ReconcileOutcome =
  | "started" // daemon was not running, we started it
  | "restarted" // extension is newer than the running daemon; stopped + started
  | "up_to_date" // versions match, nothing to do
  | "older_client" // extension is older than the running daemon; no-op
  | "failed"; // start or restart did not produce a running daemon

export interface ReconcileDeps {
  /**
   * Run `rbtr daemon start`; resolves to the status it printed, or `null`
   * when this rbtr prints none.  Throws on failure.
   */
  startDaemon: () => Promise<DaemonStatus | null>;
  /** Run `rbtr daemon stop`.  Throws on failure. */
  stopDaemon: () => Promise<void>;
}

export interface ReconcileResult {
  outcome: ReconcileOutcome;
  previousVersion: string | null;
  newVersion: string | null;
  /** On `failed`, the reason the start or stop failed. */
  detail?: string;
}

/**
 * Outcome of {@link DaemonSession.detectTransition}.  The
 * caller (health-check timer) switches on `kind` to decide
 * whether to update the footer.
 */
export type DaemonTransition = { kind: "died" } | { kind: "returned" } | { kind: "unchanged"; alive: boolean };

/** Called once per notification received on the SUB socket. */
export type NotificationHandler = (notification: Notification) => void;

export class DaemonSession {
  private status: DaemonStatus | null = null;
  private subscriber: ZmqSubscriber | null = null;
  private command: ResolvedCommand | null = null;

  /**
   * Find the daemon with *command* from now on: the session's resolved
   * `rbtr`, whose `daemon status` reported *status* (`null` if it failed).
   */
  useCommand(command: ResolvedCommand, status: DaemonStatus | null): void {
    this.command = command;
    this.status = status;
  }

  /** ``true`` if the daemon has been reachable at least once this session. */
  get available(): boolean {
    return this.status !== null && this.status.running === true;
  }

  /** Last known RPC endpoint (``ipc://...``), or ``null`` if unavailable. */
  get rpcEndpoint(): string | null {
    return this.status?.rpc ?? null;
  }

  /** Last known PUB endpoint (``ipc://...``), or ``null`` if unavailable. */
  get pubEndpoint(): string | null {
    return this.status?.pub ?? null;
  }

  /**
   * Refresh status from the CLI, run as the command given to
   * `useCommand`.  Caches the result on success.
   *
   * Any failure (no command yet, CLI not on PATH, daemon not
   * running) leaves the session marked unavailable so tools fall
   * back to CLI exec.
   */
  async refresh(): Promise<DaemonStatus | null> {
    if (this.command === null) {
      this.status = null;
      return null;
    }
    try {
      this.status = await queryDaemonStatus(this.command);
    } catch {
      this.status = null;
    }
    return this.status;
  }

  /**
   * Send a request through the cached RPC endpoint.
   *
   * A busy daemon is waited for rather than asked again (see
   * ``send``), so a failure here means the daemon is gone or
   * wedged.  The one refresh-and-retry covers the common case:
   * a daemon that restarted has a new endpoint, and the request
   * that failed was never served.  ``RbtrReplyError`` from the
   * daemon itself is propagated as-is — the caller decides what
   * to do with it, and so is an abort of *signal*: it is never
   * retried.
   *
   * A daemon known by its pid is checked before the request is sent,
   * because a request to a gone daemon is never answered.
   */
  async send<R extends Request>(request: R, signal?: AbortSignal): Promise<ResponseFor<R["kind"]>> {
    const pid = this.status?.pid;
    if (pid != null && !isAlive(pid)) this.status = null;
    if (!this.rpcEndpoint) {
      await this.refresh();
      if (!this.rpcEndpoint) {
        throw new DaemonUnavailableError("no rpc endpoint");
      }
    }

    try {
      return (await send(request, { rpcEndpoint: this.rpcEndpoint, signal })) as ResponseFor<R["kind"]>;
    } catch (err) {
      // An abort is the caller's decision, not a transport failure.
      if (signal?.aborted) throw err;
      if (err instanceof Error && err.name === "RbtrReplyError") {
        throw err;
      }
      this.status = null;
      await this.refresh();
      if (!this.rpcEndpoint) {
        throw new DaemonUnavailableError(err);
      }
      try {
        return (await send(request, { rpcEndpoint: this.rpcEndpoint, signal })) as ResponseFor<R["kind"]>;
      } catch (err2) {
        if (signal?.aborted) throw err2;
        if (err2 instanceof Error && err2.name === "RbtrReplyError") {
          throw err2;
        }
        throw new DaemonUnavailableError(err2);
      }
    }
  }

  /**
   * Re-check the daemon and classify the transition.
   *
   * A daemon known by its pid is checked with a signal-0 probe,
   * which starts no process; otherwise ``refresh()`` asks the CLI.
   * Returns a discriminated union the caller can switch on.
   */
  async detectTransition(): Promise<DaemonTransition> {
    const wasAlive = this.available;
    const pid = this.status?.pid;
    if (wasAlive && pid != null) {
      if (isAlive(pid)) return { kind: "unchanged", alive: true };
      this.status = null;
      return { kind: "died" };
    }
    await this.refresh();
    const aliveNow = this.available;

    if (wasAlive && !aliveNow) return { kind: "died" };
    if (!wasAlive && aliveNow) return { kind: "returned" };
    return { kind: "unchanged", alive: aliveNow };
  }

  /**
   * Subscribe to the daemon's PUB stream for the session.
   *
   * *handler* is called once per deserialised ``Notification``.
   * Callers typically filter on ``notification.path === cwd``
   * themselves — the daemon publishes for all watched repos.
   *
   * Returns a disposer.  Calling it (or ``stopSubscribing()``)
   * closes the SUB socket.  Safe to call twice.
   */
  subscribe(handler: NotificationHandler): () => void {
    if (this.subscriber !== null) {
      return () => {
        this.stopSubscribing();
      };
    }
    if (!this.pubEndpoint) {
      throw new DaemonUnavailableError("no pub endpoint");
    }

    const sub = new ZmqSubscriber();
    sub.connect(this.pubEndpoint);
    sub.subscribe(); // empty prefix — server publishes untopiced frames
    this.subscriber = sub;

    void (async () => {
      for await (const [frame] of sub) {
        let parsed: Notification;
        try {
          parsed = JSON.parse(frame.toString()) as Notification;
        } catch {
          continue; // malformed frame — skip
        }
        try {
          handler(parsed);
        } catch {
          // Handler errors must not kill the loop.
        }
      }
    })();

    return () => {
      this.stopSubscribing();
    };
  }

  stopSubscribing(): void {
    if (this.subscriber === null) return;
    this.subscriber.close();
    this.subscriber = null;
  }

  /**
   * Compare *expectedVersion* against the running daemon and
   * start / restart as needed so the session ends up talking to
   * a daemon at version *expectedVersion*.
   *
   * Rules:
   *   - daemon not running → start it.
   *   - running, same version → no-op.
   *   - running, extension > daemon → stop + start.  Any active
   *     build is killed; the watcher re-queues it after restart.
   *   - running, extension < daemon → no-op (older client yields).
   *
   * Versions are compared by `compareVersions`. One it cannot read
   * counts as `up_to_date`: no restart on a comparison it cannot
   * trust.
   */
  async reconcile(expectedVersion: string, deps: ReconcileDeps): Promise<ReconcileResult> {
    const before = this.status?.version ?? null;

    if (!this.available) {
      let fresh: DaemonStatus | null;
      try {
        fresh = await this.started(deps);
      } catch (err) {
        return { outcome: "failed", previousVersion: before, newVersion: null, detail: reason(err) };
      }
      return {
        outcome: fresh?.running ? "started" : "failed",
        previousVersion: before,
        newVersion: fresh?.version ?? null,
      };
    }

    const running = this.status?.version ?? null;
    const cmp = compareVersions(expectedVersion, running);
    if (cmp === null || cmp === 0) {
      return { outcome: "up_to_date", previousVersion: before, newVersion: running };
    }
    if (cmp < 0) {
      return { outcome: "older_client", previousVersion: before, newVersion: running };
    }

    // Extension is newer — restart.
    let fresh: DaemonStatus | null;
    try {
      await deps.stopDaemon();
      this.status = null;
      fresh = await this.started(deps);
    } catch (err) {
      return { outcome: "failed", previousVersion: before, newVersion: null, detail: reason(err) };
    }
    return {
      outcome: fresh?.running ? "restarted" : "failed",
      previousVersion: before,
      newVersion: fresh?.version ?? null,
    };
  }

  /**
   * Start the daemon through *deps* and cache its status: the one
   * `daemon start` printed, or, from an rbtr that prints none, a fresh
   * `daemon status`.
   */
  private async started(deps: ReconcileDeps): Promise<DaemonStatus | null> {
    const printed = await deps.startDaemon();
    if (printed === null) return this.refresh();
    this.status = printed;
    return printed;
  }
}

/** Whether process *pid* exists; `EPERM` means it does, under another user. */
function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err instanceof Error && "code" in err && err.code === "EPERM";
  }
}

/** Extract a human-readable reason from a thrown value. */
function reason(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * A release as bump-my-version writes it (`2026.9.0`, `2026.9.0-dev1`) or
 * as Python normalises it (`2026.9.0.dev1`): year, month, number, dev.
 */
const VERSION = /^(\d{4})\.(\d{1,2})\.(\d+)(?:[-.]?dev(\d+))?$/;

/** *version*'s year, month, number and dev; a release ranks above any of its dev builds. */
function versionParts(version: string): number[] | null {
  const match = VERSION.exec(version);
  if (match === null) return null;
  const [, year, month, num, dev] = match;
  return [Number(year), Number(month), Number(num), dev === undefined ? Infinity : Number(dev)];
}

/**
 * Compare two versions: negative if *a* is older than *b*, zero if they are
 * the same release, positive if *a* is newer. A missing version (`null`, a
 * daemon that reports none) is older than any; one that cannot be read
 * gives `null`.
 */
export function compareVersions(a: string | null, b: string | null): number | null {
  if (a === null || b === null) return a === b ? 0 : a === null ? -1 : 1;
  const [left, right] = [versionParts(a), versionParts(b)];
  if (left === null || right === null) return null;
  const differs = left.findIndex((part, i) => part !== right[i]);
  return differs === -1 ? 0 : Math.sign(left[differs] - right[differs]);
}
