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
from enum import StrEnum
from typing import ClassVar, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.segment import Segment
from rich.spinner import SPINNERS
from rich.table import Table
from rich.text import Text

from rbtr.config import ThinkingEffort, config
from rbtr.engine import Command, Engine, EngineState, Service, TaskType
from rbtr.engine.model import get_models
from rbtr.events import (
    CompactionFinished,
    CompactionStarted,
    Event,
    FlushPanel,
    IndexCleared,
    IndexProgress,
    IndexReady,
    IndexStarted,
    LinkOutput,
    MarkdownOutput,
    Output,
    ReviewPosted,
    TableOutput,
    TaskFinished,
    TaskStarted,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from rbtr.input import (
    InputReader,
    InputState,
    query_shell_completions,
)
from rbtr.models import PRTarget
from rbtr.styles import (
    BG_ACTIVE,
    BG_FAILED,
    BG_INPUT,
    BG_QUEUED,
    BG_SUCCEEDED,
    BG_TOOLCALL,
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
    USAGE_CRITICAL,
    USAGE_MESSAGES,
    USAGE_OK,
    USAGE_UNCERTAIN,
    USAGE_WARNING,
)
from rbtr.usage import (
    MessageCountStatus,
    ThresholdStatus,
    format_cost,
    format_tokens,
)

_SPINNER = SPINNERS["dots8"]
_SPINNER_FRAMES: list[str] = _SPINNER["frames"]  # type: ignore[assignment]  # rich Spinner dict has untyped values
_SPINNER_INTERVAL: float = _SPINNER["interval"] / 1000  # type: ignore[operator]  # rich Spinner dict has untyped values


def _format_count(n: int) -> str:
    """Format a count with k/M suffix for the footer."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


type _SegmentLine = list[Segment]


def _render_lines(console: Console, renderable: RenderableType) -> list[_SegmentLine]:
    """Render a Rich object to terminal lines at the current console width."""
    opts = console.options.update_width(console.width)
    return console.render_lines(renderable, options=opts, pad=False)


def _segment_line_to_text(line: _SegmentLine) -> Text:
    """Convert one rendered Segment line into plain Rich Text."""
    text = Text()
    for seg in line:
        if seg.control or not seg.text:
            continue
        text.append(seg.text, style=seg.style)
    return text


def _tail_renderable_lines(
    console: Console,
    renderable: RenderableType,
    max_lines: int,
) -> RenderableType:
    """Return the bottom ``max_lines`` of *renderable* for live viewporting."""
    if max_lines <= 0:
        return Text("")

    lines = _render_lines(console, renderable)
    if len(lines) <= max_lines:
        return renderable

    tail = lines[-max_lines:]
    return Group(*(_segment_line_to_text(line) for line in tail))


class _ExpandKind(StrEnum):
    """What type of content the Ctrl+O expand applies to."""

    SHELL = "shell"
    TOOL = "tool"


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
        "toolcall": BG_TOOLCALL,
    }

    def __init__(
        self,
        console: Console,
        state: EngineState,
        events: queue.Queue[Event],
        engine: Engine,
        pr_number: int | None = None,
        continue_session: bool = False,
    ) -> None:
        self.console = console
        self.state = state
        self.inp = InputState(history_provider=engine.store.search_history)
        self._events = events
        self._engine = engine
        self._pr_number = pr_number
        self._continue_session = continue_session
        self._live: Live | None = None
        # Current active panel state — built from events, never from engine state
        self._active_lines: list[RenderableType] = []
        self._streaming_text: str = ""  # accumulates TextDelta chunks
        self._streaming_md: RenderableType | None = None  # identity sentinel for replacement
        self._active_had_error = False
        self._active_task = False
        self._expandable = False  # True while Ctrl+O can expand content
        self._expand_hidden: int = 0  # number of hidden lines
        self._expand_kind: _ExpandKind | None = None  # what to expand
        self._tool_full_result: str = ""  # full result for tool expand
        self._tool_header: str = ""  # "⚙ name(args)" for tool expand
        self._tool_preamble: list[RenderableType] = []  # LLM text before tool call
        self._pending_tool_name: str = ""  # tool name from last ToolCallStarted
        self._pending_tool_args: str = ""  # tool args from last ToolCallStarted
        # Last finished panel — stays in Live until finalized by next input.
        self._pending_lines: list[RenderableType] | None = None
        self._pending_variant: Literal["succeeded", "failed", "toolcall"] = "succeeded"
        self._pending_commands: list[str] = []
        # Startup commands (e.g. /review <n>, /session resume-last) —
        # dispatched after setup, not echoed or persisted to history.
        self._startup_commands: list[str] = []
        self._needs_history_reload: bool = False
        # Index progress state — driven by IndexStarted/Progress/Ready events.
        self._index_phase: str = ""
        self._index_indexed: int = 0
        self._index_total: int = 0
        self._index_ready: bool = False
        self._index_chunks: int = 0
        # Compaction progress — set by CompactionStarted, read by CompactionFinished.
        self._compaction_old: int = 0
        self._compaction_kept: int = 0

    # ── Completion helpers ────────────────────────────────────────────

    def _complete_slash(self) -> None:
        """Generate completions for slash commands and their arguments."""
        text = self.inp.text
        parts = text.split(None, 1)
        raw_cmd = parts[0] if parts else text

        # Complete arguments for a known command
        try:
            cmd = Command(raw_cmd)
        except ValueError:
            cmd = None

        if cmd and (len(parts) == 2 or (len(parts) == 1 and text.endswith(" "))):
            arg = parts[1] if len(parts) == 2 else ""
            match cmd:
                case Command.CONNECT:
                    matches = [
                        (f"/connect {s.key}", s.description)
                        for s in Service
                        if s.key.startswith(arg)
                    ]
                    self.inp.apply_completions(matches)
                case Command.MODEL:
                    self._complete_model(arg)
                case Command.REVIEW:
                    self._complete_review(arg)
                case Command.DRAFT:
                    self._complete_draft(arg)
                case Command.INDEX:
                    subs = [
                        ("status", "Show index stats"),
                        ("clear", "Delete index"),
                        ("rebuild", "Clear and rebuild index"),
                    ]
                    matches = [
                        (f"/index {name}", desc) for name, desc in subs if name.startswith(arg)
                    ]
                    self.inp.apply_completions(matches)
                case Command.SESSION:
                    self._complete_session(arg)
            return

        # Complete the command name
        matches = [(c.slash, c.description) for c in Command if c.slash.startswith(text)]
        self.inp.apply_completions(matches)

    def _complete_model(self, partial: str) -> None:
        """Complete a model ID argument for /model.

        Uses the cached model list from ``state.cached_models``.
        If the cache is empty, fetches in a background thread.
        """
        cached = self._engine.state.cached_models
        if cached:
            flat = [m for _, models in cached for m in models]
            matches = [(f"/model {m}", "") for m in flat if m.startswith(partial)]
            self.inp.apply_completions(matches)
            return

        # No cache — fetch in background thread
        snapshot = self.inp.text

        def _run() -> None:
            if self.inp.text != snapshot:
                return
            try:
                all_models = get_models(self._engine)
            except Exception:
                return
            if self.inp.text != snapshot:
                return
            flat = [m for _, models in all_models for m in models]
            matches = [(f"/model {m}", "") for m in flat if m.startswith(partial)]
            self.inp.apply_completions(matches)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _complete_review(self, partial: str) -> None:
        """Complete a review target argument for /review.

        Uses cached PR/branch data from the last ``/review`` list.
        Falls back to local branches from the repo.
        """
        cached = self._engine.state.cached_review_targets
        if not cached and self._engine.state.repo is not None:
            # No cache yet — use local branches.
            try:
                from rbtr.git import list_local_branches

                branches = list_local_branches(self._engine.state.repo)
                cached = [(b.name, b.name) for b in branches]
            except Exception:
                cached = []
        matches = [
            (f"/review {text}", label)
            for label, text in cached
            if text.startswith(partial) or label.lower().startswith(partial.lower())
        ]
        self.inp.apply_completions(matches)

    def _complete_draft(self, partial: str) -> None:
        """Complete /draft subcommands and post event types."""
        from rbtr.engine.draft_cmd import POST_EVENTS, SUBCOMMANDS

        # "/draft post comment" → two-level completion.
        if partial.startswith("post "):
            event_partial = partial[5:]
            matches = [
                (f"/draft post {name}", desc)
                for name, desc in POST_EVENTS
                if name.startswith(event_partial)
            ]
            self.inp.apply_completions(matches)
            return

        matches = [
            (f"/draft {name}", desc) for name, desc in SUBCOMMANDS if name.startswith(partial)
        ]
        self.inp.apply_completions(matches)

    def _complete_session(self, arg: str) -> None:
        """Complete /session subcommands and session IDs for resume/delete."""
        sub_parts = arg.split(None, 1)
        subcmd = sub_parts[0] if sub_parts else arg

        # Second-level: complete session ID after "resume" or "delete".
        if subcmd in ("resume", "delete") and (
            len(sub_parts) == 2 or (len(sub_parts) == 1 and arg.endswith(" "))
        ):
            partial_id = sub_parts[1] if len(sub_parts) == 2 else ""
            sessions = self._engine.store.list_sessions(limit=50)
            current = self._engine.state.session_id
            matches = [
                (
                    f"/session {subcmd} {s.session_id[:12]}",
                    s.session_label or s.session_id[:8],
                )
                for s in sessions
                if s.session_id != current and s.session_id[:12].startswith(partial_id)
            ]
            self.inp.apply_completions(matches)
            return

        # First-level: complete subcommand name.
        subs = [
            ("all", "Sessions across all repos"),
            ("info", "Current session details"),
            ("resume", "Resume a previous session"),
            ("delete", "Delete a session by ID"),
            ("purge", "Delete sessions older than duration"),
        ]
        matches = [(f"/session {name}", desc) for name, desc in subs if name.startswith(arg)]
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
            matches = query_shell_completions(cmd_line)
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
                self._streaming_text = ""
                self._streaming_md = None
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
                    table.add_column(
                        col.header,
                        width=col.width if col.width is not None else None,
                        style=col.style or "",
                    )
                for row in te.rows:
                    table.add_row(*row)
                self._active_lines.append(table)
            case MarkdownOutput(text=text):
                self._active_lines.append(Markdown(text))
            case TextDelta(delta=delta):
                self._streaming_text += delta
                # Replace the last active line (the growing markdown)
                # with the updated accumulated text.
                if self._active_lines and self._active_lines[-1] is self._streaming_md:
                    self._active_lines[-1] = Markdown(self._streaming_text)
                else:
                    self._active_lines.append(Markdown(self._streaming_text))
                self._streaming_md = self._active_lines[-1]
            case LinkOutput(markup=markup):
                self._active_lines.append(Text.from_markup(markup))
            case FlushPanel(discard=discard):
                if not discard and self._active_lines:
                    flush_variant: Literal["succeeded", "failed"] = (
                        "failed" if self._active_had_error else "succeeded"
                    )
                    content: RenderableType = Group(*self._active_lines)
                    panel = self._history_panel(flush_variant, content)
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
                    self._print_to_scrollback(panel)
                self._active_lines = []
                self._active_had_error = False
            case ToolCallStarted(tool_name=name, args=args):
                # Don't flush — any preceding LLM explanation will be
                # merged into the same panel as the tool result.
                if self._streaming_text:
                    self._streaming_md = None
                self._pending_tool_name = name
                self._pending_tool_args = args
            case ToolCallFinished(tool_name=_name, result=result):
                # Finalize any previous pending panel (e.g. prior tool call).
                self._finalize_pending()
                # Build panel: preamble (LLM explanation) + header + result.
                preamble = list(self._active_lines)
                self._active_lines = []
                self._streaming_text = ""
                self._streaming_md = None
                args = self._pending_tool_args
                args_short = args[:80] + "…" if len(args) > 80 else args
                header_text = f"⚙ {self._pending_tool_name}({args_short})"
                header = Text(header_text, style=MUTED)
                tool_lines: list[RenderableType] = [*preamble, header]
                hidden = 0
                if result:
                    result_split = result.splitlines()
                    max_lines = config.tui.tool_max_lines
                    if len(result_split) > max_lines:
                        shown = "\n".join(result_split[:max_lines])
                        hidden = len(result_split) - max_lines
                        tool_lines.append(Text(shown, style=STYLE_DIM))
                    else:
                        tool_lines.append(Text(result, style=STYLE_DIM))
                # Keep as pending so Ctrl+O can expand if truncated.
                self._pending_lines = tool_lines
                self._pending_variant = "toolcall"
                if hidden:
                    self._expandable = True
                    self._expand_hidden = hidden
                    self._expand_kind = _ExpandKind.TOOL
                    self._tool_full_result = result
                    self._tool_header = header_text
                    self._tool_preamble = preamble
                else:
                    self._expandable = False
            case IndexStarted(total_files=total):
                self._index_phase = "parsing"
                self._index_indexed = 0
                self._index_total = total
                self._index_ready = False
            case IndexProgress(phase=phase, indexed=done, total=total):
                self._index_phase = phase
                self._index_indexed = done
                self._index_total = total
            case IndexReady(chunk_count=count):
                self._index_ready = True
                self._index_chunks = count
            case IndexCleared():
                self._index_ready = False
                self._index_chunks = 0
                self._index_phase = ""
                self._index_indexed = 0
                self._index_total = 0
            case CompactionStarted(old_messages=old, kept_messages=kept):
                self._compaction_old = old
                self._compaction_kept = kept
            case CompactionFinished(summary_tokens=tokens):
                old = self._compaction_old
                kept = self._compaction_kept
                line = Text(
                    f"Context compacted — {old} messages → summary "
                    f"(~{_format_count(tokens)} tokens) + {kept} kept",
                    style=STYLE_DIM,
                )
                panel = self._history_panel("queued", line)
                if self._live:
                    self._live.update(self._render_view(), refresh=True)
                self._print_to_scrollback(panel)
            case ReviewPosted():
                pass  # Visual feedback handled by LinkOutput
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
                variant: Literal["succeeded", "failed"] = (
                    "failed" if self._active_had_error else "succeeded"
                )
                has_active = bool(self._active_lines)
                if has_active:
                    # There's content after the last tool call (or no
                    # tool calls at all).  Finalize any pending tool
                    # panel first, then handle the active content.
                    self._finalize_pending()
                    info = self._engine._last_shell_full_output
                    if info:
                        self._expandable = True
                        self._expand_hidden = info[3]
                        self._expand_kind = _ExpandKind.SHELL
                        self._pending_lines = self._active_lines
                        self._pending_variant = variant
                    else:
                        content = Group(*self._active_lines) if self._active_lines else Text("")
                        panel = self._history_panel(variant, content)
                        if self._live:
                            self._live.update(self._render_view(), refresh=True)
                        self._print_to_scrollback(panel)
                # else: no active content — keep pending tool panel
                # as-is so the user can Ctrl+O to expand it.
                self._active_lines = []

    # ── Rendering ────────────────────────────────────────────────────

    def _render_completions(self) -> Text:
        """Render completion suggestions for commands and arguments."""
        t = Text()
        max_len = max((len(cmd) for cmd, _ in self.inp.completions), default=0)
        col_width = max(max_len + 2, 12)
        for i, (cmd, desc) in enumerate(self.inp.completions):
            if i > 0:
                t.append("\n")
            selected = i == self.inp.completion_index
            prefix = "▸ " if selected else "  "
            t.append(prefix, style=COMPLETION_SELECTED if selected else "")
            t.append(f"{cmd:<{col_width}}", style=COMPLETION_NAME)
            if desc:
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
        """Build the Live renderable — active panel + input chrome.

        The input chrome (prompt/completions/footer) is always kept
        visible by clipping oversized history content from the top.
        This preserves the *bottom* of the active/pending panel, where
        new streaming updates appear.
        """
        top_parts: list[RenderableType] = []
        if self._active_task:
            # Margin above active panel — matches the margin that
            # _print_to_scrollback adds for finalized panels.
            top_parts.append(Text(""))
            content = Group(*self._active_lines) if self._active_lines else Text("")
            top_parts.append(self._history_panel("active", content))
        elif self._pending_lines is not None:
            # Last finished panel — stays in Live so Ctrl+O can rewrite it.
            top_parts.append(Text(""))
            content = Group(*self._pending_lines) if self._pending_lines else Text("")
            bottom = 1
            if self._expandable:
                bottom = 0  # hint provides the visual closure
            top_parts.append(self._history_panel(self._pending_variant, content, bottom_pad=bottom))
            if self._expandable:
                bg = self._HISTORY_STYLES[self._pending_variant]
                hint = Text(
                    f"  … {self._expand_hidden} more lines (ctrl+o to expand)",
                    style=STYLE_DIM_ITALIC,
                )
                top_parts.append(Padding(hint, (0, 2, 1, 2), style=bg))
        if self._pending_commands:
            lines = [Text(f"  > {cmd}", style=MUTED) for cmd in self._pending_commands]
            top_parts.append(Text(""))
            top_parts.append(self._history_panel("queued", Group(*lines)))

        chrome_parts: list[RenderableType] = [Text("")]
        if self._active_task:
            frame = self._spinner_frame()
            chrome_parts.append(Rule(title=f"[{DIM}]{frame}[/{DIM}]", style=RULE))
        else:
            chrome_parts.append(Rule(style=RULE))
        chrome_parts.append(self._render_input_line())
        if self.inp.completions:
            chrome_parts.append(self._render_completions())
        chrome_parts.append(Rule(style=RULE))
        chrome_parts.append(self._render_footer())

        if not top_parts:
            return Group(*chrome_parts)

        chrome = Group(*chrome_parts)
        chrome_height = len(_render_lines(self.console, chrome))
        top_budget = self.console.height - chrome_height
        if top_budget <= 0:
            return Group(*chrome_parts)

        top = Group(*top_parts)
        clipped_top = _tail_renderable_lines(self.console, top, top_budget)
        return Group(clipped_top, *chrome_parts)

    def _spinner_frame(self) -> str:
        return _SPINNER_FRAMES[int(time.time() / _SPINNER_INTERVAL) % len(_SPINNER_FRAMES)]

    def _render_footer(self) -> Group:
        width = self.console.width

        # ── Left side ────────────────────────────────────────────────
        repo = f" {self.state.owner}/{self.state.repo_name}" if self.state.owner else " rbtr"

        target = self.state.review_target
        match target:
            case PRTarget(number=n, base_branch=base, head_branch=head):
                review = f" PR #{n} · {base} → {head}"
            case _ if target is not None:
                review = f" {target.base_branch} → {target.head_branch}"
            case _:
                review = ""
        if not self.state.gh and not review:
            review = " ✗ not authenticated"

        # Index status — appended to the review target on line 2.
        if self._index_ready:
            index_status = f" · ● {_format_count(self._index_chunks)}"
        elif self._index_total > 0:
            label = self._index_phase or "indexing"
            index_status = (
                f" · {self._spinner_frame()} {label} {self._index_indexed}/{self._index_total}"
            )
        else:
            index_status = ""
        if review and index_status:
            review += index_status

        # ── Right side ───────────────────────────────────────────────
        model = self.state.model_name or ""
        usage = self.state.usage
        has_usage = usage.input_tokens > 0 or usage.output_tokens > 0

        # Line 1: repo left, model + thinking effort right.
        # effort_supported is None until the first LLM call determines it.
        effort = config.thinking_effort
        supported = self.state.effort_supported
        if model and effort is not ThinkingEffort.NONE:
            # Show effort level; red "off" when model doesn't support it.
            if supported is False:
                effort_label = "off"
                effort_style: str | None = ERROR
            else:
                effort_label = effort
                effort_style = None
            model_right = f"{model} ∴ {effort_label} "
            line1 = Text(style=FOOTER)
            line1.append(repo)
            pad = width - len(repo) - len(model_right)
            line1.append(" " * max(pad, 2))
            line1.append(f"{model} ∴ ")
            line1.append(effort_label, style=effort_style)
            line1.append(" ")
        elif model:
            line1 = self._footer_line(repo, f"{model} ", width)
        else:
            line1 = self._footer_line(repo, "", width)

        # Single-line footer when there's nothing for line 2
        if not review and not has_usage:
            return Group(line1)

        # Line 2: review target left, usage stats right
        ctx = ""
        msgs = ""
        token_parts: list[str] = []
        if has_usage:
            msgs = f"|{usage.turn_count}:{usage.response_count}|"
            ctx_pct = f"{usage.context_used_pct:.0f}%"
            ctx_size = format_tokens(usage.context_window)
            ctx = f"{ctx_pct} of {ctx_size}"
            token_parts.append(f"↑ {format_tokens(usage.input_tokens)}")
            token_parts.append(f"↓ {format_tokens(usage.output_tokens)}")
            if usage.cache_read_tokens:
                token_parts.append(f"↯ {format_tokens(usage.cache_read_tokens)}")
            token_parts.append(format_cost(usage.total_cost))

        # Measure total width of the right side (unstyled).
        right2 = ("  ".join([msgs, ctx, *token_parts]) + " ") if has_usage else ""
        left2 = review or " "

        # Build line 2 with styled context percentage
        line2 = Text(style=FOOTER)
        line2.append(left2)
        pad = width - len(left2) - len(right2)
        line2.append(" " * max(pad, 2))

        if has_usage:
            # Message count — gray normally, yellow >25, red >50.
            match usage.message_count_status:
                case MessageCountStatus.OK:
                    msgs_style = USAGE_MESSAGES
                case MessageCountStatus.WARNING:
                    msgs_style = USAGE_WARNING
                case MessageCountStatus.CRITICAL:
                    msgs_style = USAGE_CRITICAL
            line2.append(msgs, style=msgs_style)
            line2.append("  ")

            match usage.threshold_status:
                case ThresholdStatus.OK:
                    pct_style = USAGE_OK
                case ThresholdStatus.WARNING:
                    pct_style = USAGE_WARNING
                case ThresholdStatus.CRITICAL:
                    pct_style = USAGE_CRITICAL
            # Percentage colored by threshold; total in footer color
            # (or dimmed when context window is assumed, not reported).
            line2.append(ctx_pct, style=pct_style)
            line2.append(" of ")
            total_style = USAGE_UNCERTAIN if not usage.context_window_known else None
            line2.append(format_tokens(usage.context_window), style=total_style)
            if token_parts:
                # Cost is always the last part — dimmed when unavailable.
                rest, cost = token_parts[:-1], token_parts[-1]
                if rest:
                    line2.append("  " + "  ".join(rest))
                line2.append("  ")
                cost_style = USAGE_UNCERTAIN if not usage.cost_available else None
                line2.append(cost, style=cost_style)
            line2.append(" ")
        return Group(line1, line2)

    @staticmethod
    def _footer_line(left: str, right: str, width: int) -> Text:
        """Build a single footer line with left/right alignment."""
        t = Text(style=FOOTER)
        t.append(left)
        pad = width - len(left) - len(right)
        t.append(" " * max(pad, 2))
        if right:
            t.append(right)
        return t

    def _history_panel(
        self,
        variant: Literal["input", "active", "succeeded", "failed", "queued", "toolcall"],
        content: RenderableType,
        *,
        bottom_pad: int = 1,
    ) -> Padding:
        """Build a history block — background band, no borders.

        Uses background colour for visual grouping (copy-paste friendly).
        bottom_pad=0 when a continuation (hint/expand) will follow.
        """
        bg = self._HISTORY_STYLES[variant]
        return Padding(content, (1, 2, bottom_pad, 2), style=bg)

    def _print_to_scrollback(self, renderable: RenderableType, *, margin_top: bool = True) -> None:
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
        self._expand_kind = None
        self._engine._last_shell_full_output = None
        if self._live:
            self._live.update(self._render_view(), refresh=True)
        self._print_to_scrollback(panel)

    def _reload_history(self) -> None:
        """Reload up-arrow history from the DB."""
        provider = self.inp.history_provider
        if provider is not None:
            self.inp.history = list(reversed(provider(None, config.tui.max_history)))

    def _echo_input(self, text: str) -> None:
        """Print an Input HistoryPanel to native scrollback."""
        self._finalize_pending()
        t = Text()
        t.append("> ", style=PROMPT)
        t.append(text, style=INPUT_TEXT)
        self._print_to_scrollback(self._history_panel("input", t))

    def _expand_last_output(self) -> None:
        """Replace the pending panel's truncated content with full output."""
        if not self._expandable:
            return
        match self._expand_kind:
            case _ExpandKind.SHELL:
                self._expand_shell()
            case _ExpandKind.TOOL:
                self._expand_tool()

    def _expand_shell(self) -> None:
        """Expand a truncated shell output panel."""
        if self._engine._last_shell_full_output is None:
            return
        stdout_full, stderr_full, returncode, _ = self._engine._last_shell_full_output
        self._engine._last_shell_full_output = None
        self._expandable = False
        self._expand_kind = None
        # Rebuild content: keep the "$ command" echo (first line),
        # replace truncated output with full output.
        echo = self._pending_lines[0] if self._pending_lines else None
        expanded: list[RenderableType] = []
        if echo is not None:
            expanded.append(echo)
        if stdout_full:
            expanded.append(Text(stdout_full, style=STYLE_DIM))
        if stderr_full:
            expanded.append(Text(stderr_full, style=STYLE_SHELL_STDERR))
        if returncode != 0:
            t = Text()
            t.append("Error: ", style=ERROR)
            t.append(f"(exit code {returncode})")
            expanded.append(t)
        self._pending_lines = expanded
        # Expanded panel is finalized — move to scrollback so the full
        # output is scrollable and Live shrinks back to input chrome.
        self._finalize_pending()

    def _expand_tool(self) -> None:
        """Expand a truncated tool call output panel."""
        if not self._tool_full_result:
            return
        self._expandable = False
        self._expand_kind = None
        self._pending_lines = [
            *self._tool_preamble,
            Text(self._tool_header, style=MUTED),
            Text(self._tool_full_result, style=STYLE_DIM),
        ]
        self._tool_full_result = ""
        self._tool_header = ""
        self._tool_preamble = []
        self._finalize_pending()

    # ── Effort rotation ─────────────────────────────────────────────

    @staticmethod
    def _rotate_effort() -> None:
        """Cycle thinking effort: low → medium → high → max → none → low."""
        members = list(ThinkingEffort)
        current = config.thinking_effort
        idx = members.index(current)
        config.update(thinking_effort=members[(idx + 1) % len(members)])

    # ── Task dispatch ────────────────────────────────────────────────

    def _start_task(self, task_type: TaskType, arg: str, *, persist: bool = True) -> None:
        """Start a task in a daemon thread."""
        # Clear cancel from any previous task, then mark active
        # immediately so the input reader sees it before the thread
        # emits TaskStarted — avoids a race where Ctrl-C right after
        # dispatch clears the input instead of cancelling.
        self._engine._cancel.clear()
        self._active_task = True
        self.inp.active_task = True
        t = threading.Thread(
            target=self._engine.run_task,
            args=(task_type, arg),
            kwargs={"persist": persist},
            daemon=True,
        )
        t.start()

    def _dispatch(self, raw: str) -> None:
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            try:
                cmd = Command(parts[0].lower())
            except ValueError:
                cmd = None
            if cmd is Command.QUIT:
                self._engine._persist_input(raw, "command")
                self.inp.quit = True
                return
            self._start_task(TaskType.COMMAND, raw)
        elif raw.startswith("!"):
            self._start_task(TaskType.SHELL, raw[1:].strip())
        else:
            self._start_task(TaskType.LLM, raw)

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        with (
            InputReader(self.inp) as _reader,
            Live(
                self._render_view(),
                console=self.console,
                refresh_per_second=config.tui.refresh_per_second,
                transient=True,
            ) as live,
        ):
            self._live = live
            self.inp.on_cancel = self._engine.cancel
            self._start_task(TaskType.SETUP, "")
            if self._pr_number is not None:
                self._startup_commands.append(f"/review {self._pr_number}")
            elif self._continue_session:
                self._startup_commands.append("/session resume-last")

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

                # Handle Shift+Tab → rotate thinking effort level
                if self.inp.shift_tab_pressed:
                    self.inp.shift_tab_pressed = False
                    self._rotate_effort()
                    live.update(self._render_view())

                # Process startup commands (no echo, no history).
                if not self._active_task and self._startup_commands:
                    cmd = self._startup_commands.pop(0)
                    self._needs_history_reload = True
                    self._start_task(TaskType.COMMAND, cmd, persist=False)
                    live.update(self._render_view())

                # Reload history from DB after startup commands finish
                # so resumed session commands appear in up-arrow.
                if self._needs_history_reload and not self._active_task:
                    self._needs_history_reload = False
                    self._reload_history()

                # Process queued commands when task finishes
                if not self._active_task and self._pending_commands:
                    cmd = self._pending_commands.pop(0)
                    self._dispatch(cmd)
                    live.update(self._render_view())

                # Handle expand request
                if self.inp.expand_requested:
                    self.inp.expand_requested = False
                    self._expand_last_output()
                    live.update(self._render_view())

                # Process submitted input
                try:
                    raw = self.inp.submitted.get(timeout=config.tui.poll_interval)
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


def run(
    *,
    pr_number: int | None = None,
    continue_session: bool = False,
) -> None:
    """Launch the rbtr interactive session."""
    console = Console(markup=True, highlight=False, theme=THEME)
    state = EngineState()
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(state, events)
    try:
        ui = UI(
            console,
            state,
            events,
            engine,
            pr_number=pr_number,
            continue_session=continue_session,
        )
        ui.run()
    finally:
        engine.close()
