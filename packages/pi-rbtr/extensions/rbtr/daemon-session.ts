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
import type { Notification, Request, Response } from "./generated/protocol.js";

/**
 * A transport-level failure talking to the daemon — socket
 * refused, send/recv timeout, JSON parse.  Distinct from
 * ``RbtrDaemonError`` which means the daemon replied with a
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
  /** Run `rbtr daemon start` / `rbtr daemon stop`. */
  execDaemon: (subcommand: "start" | "stop") => Promise<void>;
}

export interface ReconcileResult {
  outcome: ReconcileOutcome;
  previousVersion: string | null;
  newVersion: string | null;
  /** On `failed`, the reason from the failing `execDaemon` call. */
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
   * Refresh status from the CLI.  Caches the result on success.
   *
   * Any failure (CLI not on PATH, daemon not running) leaves the
   * session marked unavailable so tools fall back to CLI exec.
   */
  async refresh(): Promise<DaemonStatus | null> {
    try {
      this.status = await queryDaemonStatus();
    } catch {
      this.status = null;
    }
    return this.status;
  }

  /**
   * Send a request through the cached RPC endpoint.
   *
   * Re-queries once on transport failure (covers daemon restart
   * with a new endpoint) before giving up with
   * ``DaemonUnavailableError``.  ``RbtrDaemonError`` from the
   * daemon itself is propagated as-is — the caller decides what
   * to do with it.
   */
  async send<R extends Request>(request: R): Promise<ResponseFor<R["kind"]>> {
    if (!this.rpcEndpoint) {
      await this.refresh();
      if (!this.rpcEndpoint) {
        throw new DaemonUnavailableError("no rpc endpoint");
      }
    }

    try {
      return (await send(request, { rpcEndpoint: this.rpcEndpoint })) as ResponseFor<R["kind"]>;
    } catch (err) {
      if (err instanceof Error && err.name === "RbtrDaemonError") {
        throw err;
      }
      // Transport failure — try one refresh + retry before
      // giving up.  Covers daemon restart with a new endpoint.
      this.status = null;
      await this.refresh();
      if (!this.rpcEndpoint) {
        throw new DaemonUnavailableError(err);
      }
      try {
        return (await send(request, { rpcEndpoint: this.rpcEndpoint })) as ResponseFor<R["kind"]>;
      } catch (err2) {
        if (err2 instanceof Error && err2.name === "RbtrDaemonError") {
          throw err2;
        }
        throw new DaemonUnavailableError(err2);
      }
    }
  }

  /**
   * Refresh daemon status and classify the transition.
   *
   * Captures ``available`` before calling ``refresh()``,
   * compares with the post-refresh value, and returns a
   * discriminated union the caller can switch on.
   */
  async detectTransition(): Promise<DaemonTransition> {
    const wasAlive = this.available;
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
   * Version strings are compared lexicographically — both sides
   * use calver (`YYYY.MM.PATCH`) so string compare agrees with
   * time order.  Unparseable strings fall through as `up_to_date`
   * (conservative: don't restart on a comparison we can't trust).
   */
  async reconcile(expectedVersion: string, deps: ReconcileDeps): Promise<ReconcileResult> {
    const before = this.status?.version ?? null;

    if (!this.available) {
      try {
        await deps.execDaemon("start");
      } catch (err) {
        return { outcome: "failed", previousVersion: before, newVersion: null, detail: reason(err) };
      }
      const fresh = await this.refresh();
      return {
        outcome: fresh?.running ? "started" : "failed",
        previousVersion: before,
        newVersion: fresh?.version ?? null,
      };
    }

    const running = this.status?.version ?? null;
    const cmp = compareCalver(expectedVersion, running);
    if (cmp === 0) {
      return { outcome: "up_to_date", previousVersion: before, newVersion: running };
    }
    if (cmp < 0) {
      return { outcome: "older_client", previousVersion: before, newVersion: running };
    }

    // Extension is newer — restart.
    try {
      await deps.execDaemon("stop");
      await deps.execDaemon("start");
    } catch (err) {
      return { outcome: "failed", previousVersion: before, newVersion: null, detail: reason(err) };
    }
    this.status = null;
    const fresh = await this.refresh();
    return {
      outcome: fresh?.running ? "restarted" : "failed",
      previousVersion: before,
      newVersion: fresh?.version ?? null,
    };
  }
}

/**
 * Compare two calver strings lexicographically.  `null` sorts below
 * any string (used when the daemon has no version to report).
 *
 * Returns a negative number if *a* < *b*, zero if equal, positive
 * if *a* > *b*.
 */
/** Extract a human-readable reason from a thrown value. */
function reason(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function compareCalver(a: string | null, b: string | null): number {
  if (a === b) return 0;
  if (a === null) return -1;
  if (b === null) return 1;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}
