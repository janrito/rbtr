"""Interactive CLI for rbtr, powered entirely by Rich.

Architecture: execution and display are fully separated.

**Engine** (``rbtr.engine``, daemon threads): runs commands, produces typed
Events onto a queue. Knows nothing about Rich, Live, or Panels. Never
touches the display.

**UI** (this module, main thread): owns the Rich Live context, consumes
Events, renders. Never runs commands or does I/O beyond rendering.

They communicate through ``queue.Queue[Event]`` using Pydantic models
defined in ``rbtr.events``.

The terminal's native scroll buffer holds all history. The Live region
is kept small — only the current panel + input chrome.
"""

import queue
import threading
import time
from typing import ClassVar, Literal

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.spinner import SPINNERS
from rich.table import Table
from rich.text import Text

from rbtr.constants import (
    POLL_INTERVAL,
    REFRESH_PER_SECOND,
    SHELL_MAX_COMPLETIONS,
)
from rbtr.engine import _COMMANDS, Engine, Session
from rbtr.events import (
    Event,
    FlushPanel,
    LinkOutput,
    MarkdownOutput,
    Output,
    TableOutput,
    TaskFinished,
    TaskStarted,
)
from rbtr.input import (
    InputReader,
    InputState,
    query_shell_completions,
)
from rbtr.styles import (
    BG_ACTIVE,
    BG_FAILED,
    BG_INPUT,
    BG_QUEUED,
    BG_SUCCEEDED,
    COMPLETION_DESC,
    COMPLETION_NAME,
    COMPLETION_SELECTED,
    CURSOR,
    DIM,
    ERROR,
    FOOTER,
    INPUT_TEXT,
    MUTED,
    PROMPT,
    RULE,
    STYLE_DIM,
    STYLE_DIM_ITALIC,
    STYLE_SHELL_STDERR,
    STYLE_WARNING,
    THEME,
)

_SPINNER = SPINNERS["dots8"]
_SPINNER_FRAMES: str = _SPINNER["frames"]
_SPINNER_INTERVAL: float = _SPINNER["interval"] / 1000  # ms → seconds


# ═══════════════════════════════════════════════════════════════════════
# UI — owns the terminal, consumes Events, renders
# ═══════════════════════════════════════════════════════════════════════


class UI:
    """Owns the Rich Live display and input handling. Consumes engine events."""

    _HISTORY_STYLES: ClassVar[dict[str, str]] = {
        "input": BG_INPUT,
        "active": BG_ACTIVE,
        "succeeded": BG_SUCCEEDED,
        "failed": BG_FAILED,
        "queued": BG_QUEUED,
    }

    def __init__(
        self,
        console: Console,
        session: Session,
        events: queue.Queue[Event],
        engine: Engine,
        pr_number: int | None = None,
    ) -> None:
        self.console = console
        self.session = session
        self.inp = InputState()
        self._events = events
        self._engine = engine
        self._pr_number = pr_number
        self._live: Live | None = None
        # Current active panel state — built from events, never from engine state
        self._active_lines: list[object] = []
        self._active_had_error = False
        self._active_task = False
        self._expandable = False  # True while Ctrl+O can expand last shell output
        self._expand_hidden: int = 0  # number of hidden lines
        # Last finished panel — stays in Live until finalized by next input.
        self._pending_lines: list[object] | None = None
        self._pending_variant: Literal["succeeded", "failed"] = "succeeded"
        self._pending_commands: list[str] = []

    # ── Completion helpers ────────────────────────────────────────────

    def _complete_slash(self) -> None:
        """Generate completions for slash commands."""
        matches = [(c, _COMMANDS[c]) for c in _COMMANDS if c.startswith(self.inp.text)]
        self.inp.apply_completions(matches)

    def _complete_shell(self) -> None:
        """Complete a shell command in a background thread.

        Delegates to ``query_shell_completions`` (bash completion →
        filesystem → PATH search) and applies results to input state.
        Runs off the main thread so the UI stays responsive.
        """
        cmd_line = self.inp.text[1:]  # strip leading !
        if not cmd_line:
            return

        snapshot = self.inp.text

        def _run() -> None:
            if self.inp.text != snapshot:
                return
            matches = query_shell_completions(cmd_line, SHELL_MAX_COMPLETIONS)
            if self.inp.text != snapshot:
                return
            self.inp.apply_completions(matches)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── Event consumer ───────────────────────────────────────────────

    def _drain_events(self) -> None:
        """Consume all pending events from the engine and update UI state."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)

    def _handle_event(self, event: Event) -> None:
        """Process a single engine event."""
        match event:
            case TaskStarted():
                self._expandable = False
                self._active_lines.clear()
                self._active_had_error = False
                self._active_task = True
            case Output(text=text, style=style):
                if "error" in style:
                    self._active_had_error = True
                    t = Text()
                    t.append("Error: ", style=ERROR)
                    t.append(text)
                    self._active_lines.append(t)
                else:
                    self._active_lines.append(Text(text, style=style))
            case TableOutput() as te:
                table = Table(title=te.title, show_lines=False, style=te.style)
                for col in te.columns:
                    kwargs: dict[str, object] = {}
                    if col.width is not None:
                        kwargs["width"] = col.width
                    if col.style:
                        kwargs["style"] = col.style
                    table.add_column(col.header, **kwargs)
                for row in te.rows:
                    table.add_row(*row)
                self._active_lines.append(table)
            case MarkdownOutput(text=text):
                self._active_lines.append(Markdown(text))
            case LinkOutput(markup=markup):
                self._active_lines.append(Text.from_markup(markup))
            case FlushPanel(discard=discard):
                if not discard and self._active_lines:
                    variant: Literal["succeeded", "failed"] = (
                        "failed" if self._active_had_error else "succeeded"
                    )
                    content = Group(*self._active_lines)
                    panel = self._history_panel(variant, content)
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
                    self._print_to_scrollback(panel)
                self._active_lines = []
                self._active_had_error = False
            case TaskFinished(success=success, cancelled=cancelled):
                if not success:
                    self._active_had_error = True
                if cancelled:
                    # Remove "Cancelling…" (may not exist if the cancel
                    # was faster than the main loop's polling cycle)
                    # and always add "Cancelled."
                    self._active_lines = [
                        line
                        for line in self._active_lines
                        if not (isinstance(line, Text) and line.plain == "Cancelling…")
                    ]
                    self._active_lines.append(Text("Cancelled.", style=STYLE_WARNING))
                    self._pending_commands.clear()
                self._active_task = False
                variant = (
                    "failed" if self._active_had_error else "succeeded"
                )
                info = self._engine._last_shell_full_output
                self._expandable = info is not None
                if info:
                    self._expand_hidden = info[3]
                # Finalize any previous pending panel first.
                self._finalize_pending()
                if self._expandable:
                    # Output was truncated — keep in Live so Ctrl+O
                    # can expand it before it goes to scrollback.
                    self._pending_lines = self._active_lines
                    self._pending_variant = variant
                else:
                    # All output fits — send straight to scrollback
                    # so it's immediately selectable/copyable.
                    content = Group(*self._active_lines) if self._active_lines else Text("")
                    panel = self._history_panel(variant, content)
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
                    self._print_to_scrollback(panel)
                self._active_lines = []

    # ── Rendering ────────────────────────────────────────────────────

    def _render_completions(self) -> Text:
        """Render slash command completion suggestions."""
        t = Text()
        for i, (cmd, desc) in enumerate(self.inp.completions):
            if i > 0:
                t.append("\n")
            selected = i == self.inp.completion_index
            prefix = "▸ " if selected else "  "
            t.append(prefix, style=COMPLETION_SELECTED if selected else "")
            t.append(f"{cmd:<12}", style=COMPLETION_NAME)
            t.append(desc, style=COMPLETION_DESC)
        return t

    def _render_input_line(self) -> Text:
        t = Text()
        t.append("> ", style=PROMPT)
        buf = self.inp.text
        pos = self.inp.cursor
        t.append(buf[:pos])
        # Block cursor: reverse-video the character under the cursor,
        # same as a real terminal block cursor.
        char = buf[pos] if pos < len(buf) else " "
        t.append(char, style=CURSOR)
        t.append(buf[pos + 1 :])
        return t

    def _render_view(self) -> Group:
        """Build the Live renderable — active panel + input chrome."""
        parts: list[object] = []
        if self._active_task:
            # Margin above active panel — matches the margin that
            # _print_to_scrollback adds for finalized panels.
            parts.append(Text(""))
            content = Group(*self._active_lines) if self._active_lines else Text("")
            parts.append(self._history_panel("active", content))
        elif self._pending_lines is not None:
            # Last finished panel — stays in Live so Ctrl+O can rewrite it.
            parts.append(Text(""))
            content = Group(*self._pending_lines) if self._pending_lines else Text("")
            bottom = 1
            if self._expandable:
                bottom = 0  # hint provides the visual closure
            parts.append(self._history_panel(self._pending_variant, content, bottom_pad=bottom))
            if self._expandable:
                bg = self._HISTORY_STYLES[self._pending_variant]
                hint = Text(
                    f"  … {self._expand_hidden} more lines (ctrl+o to expand)",
                    style=STYLE_DIM_ITALIC,
                )
                parts.append(Padding(hint, (0, 2, 1, 2), style=bg))
        if self._pending_commands:
            lines = [Text(f"  > {cmd}", style=MUTED) for cmd in self._pending_commands]
            parts.append(Text(""))
            parts.append(self._history_panel("queued", Group(*lines)))
        parts.append(Text(""))
        if self._active_task:
            frame = self._spinner_frame()
            parts.append(Rule(title=f"[{DIM}]{frame}[/{DIM}]", style=RULE))
        else:
            parts.append(Rule(style=RULE))
        parts.append(self._render_input_line())
        if self.inp.completions:
            parts.append(self._render_completions())
        parts.append(Rule(style=RULE))
        parts.append(self._render_footer())
        return Group(*parts)

    def _spinner_frame(self) -> str:
        return _SPINNER_FRAMES[int(time.time() / _SPINNER_INTERVAL) % len(_SPINNER_FRAMES)]

    def _footer_text(self) -> str:
        parts: list[str] = []
        if self.session.owner:
            parts.append(f"{self.session.owner}/{self.session.repo_name}")
        if self.session.pr is not None:
            if self.session.pr.number is not None:
                parts.append(f"PR #{self.session.pr.number} · {self.session.pr.head_branch}")
            else:
                parts.append(self.session.pr.head_branch)
        if self.session.gh is None:
            parts.append("✗ not authenticated")
        return " │ ".join(parts) if parts else "rbtr"

    def _render_footer(self) -> Text:
        return Text(f" {self._footer_text()}", style=FOOTER)

    def _history_panel(
        self,
        variant: Literal["input", "active", "succeeded", "failed", "queued"],
        content: object,
        *,
        bottom_pad: int = 1,
    ) -> Padding:
        """Build a history block — background band, no borders.

        Uses background colour for visual grouping (copy-paste friendly).
        bottom_pad=0 when a continuation (hint/expand) will follow.
        """
        bg = self._HISTORY_STYLES[variant]
        return Padding(content, (1, 2, bottom_pad, 2), style=bg)

    def _print_to_scrollback(self, renderable: object, *, margin_top: bool = True) -> None:
        """Print to the terminal's native scrollback.

        margin_top: print an empty line before the renderable so adjacent
        panels have visual separation *outside* the bg band.
        """
        console = self._live.console if self._live else self.console
        if margin_top:
            console.print()
        console.print(renderable)

    def _finalize_pending(self) -> None:
        """Print the pending panel to scrollback and clear it.

        If the panel was expandable but not expanded, the hint is baked
        into the content as "… N more lines" (no ctrl+o).
        """
        if self._pending_lines is None:
            return
        lines = list(self._pending_lines)
        if self._expandable:
            lines.append(Text(f"  … {self._expand_hidden} more lines", style=STYLE_DIM_ITALIC))
        content = Group(*lines) if lines else Text("")
        panel = self._history_panel(self._pending_variant, content)
        # Clear pending BEFORE printing so Live shrinks to input chrome
        # first — otherwise the large panel in Live pushes input off-screen.
        self._pending_lines = None
        self._expandable = False
        self._engine._last_shell_full_output = None
        if self._live:
            self._live.update(self._render_view(), refresh=True)
        self._print_to_scrollback(panel)

    def _echo_input(self, text: str) -> None:
        """Print an Input HistoryPanel to native scrollback."""
        self._finalize_pending()
        t = Text()
        t.append("> ", style=PROMPT)
        t.append(text, style=INPUT_TEXT)
        self._print_to_scrollback(self._history_panel("input", t))

    def _expand_last_output(self) -> None:
        """Replace the pending panel's truncated content with full output."""
        if not self._expandable or self._engine._last_shell_full_output is None:
            return
        stdout_full, stderr_full, returncode, _ = self._engine._last_shell_full_output
        self._engine._last_shell_full_output = None
        self._expandable = False
        # Rebuild content: keep the "$ command" echo (first line),
        # replace truncated output with full output.
        echo = self._pending_lines[0] if self._pending_lines else None
        lines: list[object] = []
        if echo is not None:
            lines.append(echo)
        if stdout_full:
            lines.append(Text(stdout_full, style=STYLE_DIM))
        if stderr_full:
            lines.append(Text(stderr_full, style=STYLE_SHELL_STDERR))
        if returncode != 0:
            t = Text()
            t.append("Error: ", style=ERROR)
            t.append(f"(exit code {returncode})")
            lines.append(t)
        self._pending_lines = lines
        # Expanded panel is finalized — move to scrollback so the full
        # output is scrollable and Live shrinks back to input chrome.
        self._finalize_pending()

    # ── Task dispatch ────────────────────────────────────────────────

    def _start_task(self, task_type: str, arg: str) -> None:
        """Start a task in a daemon thread."""
        # Clear cancel from any previous task, then mark active
        # immediately so the input reader sees it before the thread
        # emits TaskStarted — avoids a race where Ctrl-C right after
        # dispatch clears the input instead of cancelling.
        self._engine._cancel.clear()
        self._active_task = True
        self.inp.active_task = True
        t = threading.Thread(target=self._engine.run_task, args=(task_type, arg), daemon=True)
        t.start()

    def _dispatch(self, raw: str) -> None:
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            if cmd in ("/quit", "/q"):
                self.inp.quit = True
                return
            self._start_task("command", raw)
        elif raw.startswith("!"):
            self._start_task("shell", raw[1:].strip())
        else:
            self._start_task("llm", raw)

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        with (
            InputReader(self.inp) as _reader,
            Live(
                self._render_view(),
                console=self.console,
                refresh_per_second=REFRESH_PER_SECOND,
                transient=True,
            ) as live,
        ):
            self._live = live
            self.inp.on_cancel = self._engine.cancel
            self._start_task("setup", "")
            if self._pr_number is not None:
                self._pending_commands.append(f"/review {self._pr_number}")
            else:
                self._pending_commands.append("/review")

            while not self.inp.quit:
                # Consume engine events and update UI state
                self._drain_events()

                # Keep the reader informed about task state.
                self.inp.active_task = self._active_task

                # Handle cancel request from input thread.
                # engine.cancel() is already called directly from
                # the reader thread (on_cancel callback) for immediacy.
                # Here we do the bookkeeping: visual feedback + cleanup.
                if self.inp.cancel_requested:
                    self.inp.cancel_requested = False
                    self._pending_commands.clear()
                    if self._active_task:
                        self._active_lines.append(Text("Cancelling…", style=STYLE_DIM_ITALIC))

                # Handle tab completion request from input thread
                if self.inp.tab_pressed:
                    self.inp.tab_pressed = False
                    if self.inp.text.startswith("/"):
                        self._complete_slash()
                    elif self.inp.text.startswith("!") and len(self.inp.text) > 1:
                        self._complete_shell()

                # Process queued commands when task finishes
                if not self._active_task and self._pending_commands:
                    cmd = self._pending_commands.pop(0)
                    self.inp.append_history(cmd)
                    self._echo_input(cmd)
                    self._dispatch(cmd)
                    live.update(self._render_view())

                # Handle expand request
                if self.inp.expand_requested:
                    self.inp.expand_requested = False
                    self._expand_last_output()
                    live.update(self._render_view())

                # Process submitted input
                try:
                    raw = self.inp.submitted.get(timeout=POLL_INTERVAL)
                except queue.Empty:
                    live.update(self._render_view())
                    continue

                if raw is None:
                    break

                if self._active_task:
                    self._pending_commands.append(raw)
                else:
                    self._echo_input(raw)
                    self._dispatch(raw)

                live.update(self._render_view())

            self._live = None


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════


def run(pr_number: int | None) -> None:
    """Launch the rbtr interactive session."""
    console = Console(markup=True, highlight=False, theme=THEME)
    session = Session()
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(session, events, pr_number=pr_number)
    ui = UI(console, session, events, engine, pr_number)
    ui.run()
