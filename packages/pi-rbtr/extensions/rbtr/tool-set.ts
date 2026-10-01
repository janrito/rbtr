/**
 * Which rbtr tools the model starts with.
 *
 * The tools that find code, and `rbtr_watch`, are active from the start:
 * asking for refs to be indexed before a review is an ordinary request.
 * `rbtr_status` and `rbtr_gc` wait until the model calls the loader,
 * `rbtr_index_tools`, so the prompt the model reads first is about
 * finding code.
 */

export const LOADER = "rbtr_index_tools";

export const ON_REQUEST: readonly string[] = ["rbtr_status", "rbtr_gc"];

/** The active tools at session start: all of them but the ones loaded on request. */
export function startingTools(active: readonly string[]): string[] {
  return active.filter((name) => !ON_REQUEST.includes(name));
}

/** The active tools with the ones loaded on request added. */
export function withOnRequest(active: readonly string[]): string[] {
  return [...active, ...ON_REQUEST.filter((name) => !active.includes(name))];
}
