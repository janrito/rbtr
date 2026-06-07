/**
 * Scenarios for {@link DaemonSession.detectTransition}.
 *
 * Data-first: each case names a behaviour, spells out the
 * session's initial state, what `refresh()` returns, and the
 * expected transition variant.
 */

export interface TransitionScenario {
  /** Human-readable name (used as the test case ID). */
  readonly name: string;
  /** `session.available` before `refresh()`. */
  readonly initialAvailable: boolean;
  /** What `refresh()` sets `session.available` to. */
  readonly refreshedAvailable: boolean;
  /** Expected discriminant on the returned `DaemonTransition`. */
  readonly expectedKind: "died" | "returned" | "unchanged";
}

export const transitionScenarios: readonly TransitionScenario[] = [
  {
    name: "daemon dies (was alive, now dead)",
    initialAvailable: true,
    refreshedAvailable: false,
    expectedKind: "died",
  },
  {
    name: "daemon returns (was dead, now alive)",
    initialAvailable: false,
    refreshedAvailable: true,
    expectedKind: "returned",
  },
  {
    name: "daemon stays alive",
    initialAvailable: true,
    refreshedAvailable: true,
    expectedKind: "unchanged",
  },
  {
    name: "daemon stays dead",
    initialAvailable: false,
    refreshedAvailable: false,
    expectedKind: "unchanged",
  },
];
