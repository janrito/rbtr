/**
 * Footer status helper with animated spinner support.
 *
 * The daemon emits progress notifications sporadically — sometimes
 * with long silent gaps (model load, checkpoint, orphan sweep).
 * A static glyph in the footer reads as "frozen".  This module
 * owns a single timer that rotates through braille spinner frames
 * so the footer always has visible motion while a build is live.
 * Only in the terminal UI: in RPC mode every status is a message on
 * stdout, so there the spinner is painted once, with a static glyph.
 * With no UI (print and JSON modes, an SDK host that passes none) it
 * paints nothing: there is no footer, and pi's theme may not exist.
 */

import type { ExtensionContext, Theme, ThemeColor } from "@earendil-works/pi-coding-agent";

/** The spinner's frames in the terminal UI, in order. */
export const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_INTERVAL_MS = 100;
/** The spinner's one frame where it does not animate. */
export const STATIC_FRAME = "⟳";

/**
 * Render the given *body* against the current spinner frame, tinted
 * in `color`.  The *body* is expected to start with a leading
 * placeholder for the spinner glyph — callers pass ``"{spinner}"``.
 */
type Render = (frame: string) => string;

export class Footer {
  private timer: ReturnType<typeof setInterval> | null = null;
  private frameIdx = 0;
  private readonly animate: boolean;
  private readonly paints: boolean;

  constructor(
    private readonly ctx: ExtensionContext,
    private readonly key: string = "rbtr",
  ) {
    this.animate = ctx.mode === "tui";
    this.paints = ctx.hasUI;
  }

  /** Clear any running spinner and paint *text* in *color*. */
  setStatic(color: ThemeColor, text: string): void {
    this.stopSpinner();
    if (!this.paints) return;
    this.ctx.ui.setStatus(this.key, this.ctx.ui.theme.fg(color, text));
  }

  /**
   * Start an animated spinner.  The *render* callback is called
   * once per frame with the current spinner glyph and must return
   * the full footer string (including the prefix ``"rbtr: "``).
   * The string is tinted with *color*.
   *
   * If a spinner is already running, it is replaced — the
   * animation continues without flicker.
   */
  setSpinner(color: ThemeColor, render: Render): void {
    this.stopSpinner();
    if (!this.paints) return;
    if (!this.animate) {
      this.ctx.ui.setStatus(this.key, this.ctx.ui.theme.fg(color, render(STATIC_FRAME)));
      return;
    }
    const paint = (): void => {
      const frame = SPINNER_FRAMES[this.frameIdx];
      this.frameIdx = (this.frameIdx + 1) % SPINNER_FRAMES.length;
      this.ctx.ui.setStatus(this.key, this.ctx.ui.theme.fg(color, render(frame)));
    };
    paint();
    this.timer = setInterval(paint, SPINNER_INTERVAL_MS);
  }

  private stopSpinner(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  /** Tear down the timer; safe to call more than once. */
  dispose(): void {
    this.stopSpinner();
  }
}

/** Expose the theme color palette to callers who already have a Theme. */
export type { Theme, ThemeColor };
