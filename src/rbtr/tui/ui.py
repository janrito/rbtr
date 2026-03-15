"""Interactive CLI for rbtr, powered entirely by Rich.

Architecture: execution and display are fully separated.

**Engine** (`rbtr.engine`, daemon threads): runs commands, produces typed
Events onto a queue. Knows nothing about Rich, Live, or Panels. Never
touches the display.

**UI** (this module, main thread): owns the Rich Live context, consumes
Events, renders. Never runs commands or does I/O beyond rendering.

They communicate through `queue.Queue[Event]` using Pydantic models
defined in `rbtr.events`.

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
from rbtr.engine import Command, Engine, Service, TaskType
from rbtr.engine.model_cmd import get_models
from rbtr.events import (
    CompactionFinished,
    CompactionStarted,
    ContextMarkerReady,
    Event,
    FactExtractionFinished,
    FactExtractionStarted,
    FlushPanel,
    IndexCleared,
    IndexProgress,
    IndexReady,
    IndexStarted,
    LinkOutput,
    MarkdownOutput,
    Output,
    OutputLevel,
    ReviewPosted,
    TableOutput,
    TaskFinished,
    TaskStarted,
    TextDelta,
    ToolCallFinished,
    ToolCallOutput,
    ToolCallStarted,
)
from rbtr.providers import PROVIDERS
from rbtr.state import EngineState
from rbtr.styles import (
    BG_ACTIVE,
    BG_FAILED,
    BG_INPUT,
    BG_QUEUED,
    BG_SUCCEEDED,
    BG_TOOLCALL,
    COLUMN_BRANCH,
    COMPLETION_DESC,
    COMPLETION_NAME,
    COMPLETION_SELECTED,
    CONTEXT_MARKER,
    CURSOR,
    DIM,
    ERROR,
    INPUT_TEXT,
    LINK_STYLE,
    MUTED,
    PASTE_MARKER,
    PROMPT,
    RULE,
    STYLE_DIM,
    STYLE_DIM_ITALIC,
    STYLE_ERROR,
    STYLE_SHELL_STDERR,
    STYLE_WARNING,
    build_theme,
)
from rbtr.tui.footer import _format_count, render_footer
from rbtr.tui.input import (
    InputReader,
    InputState,
    MarkerKind,
    PasteRegion,
    query_shell_completions,
)

_SPINNER = SPINNERS["dots8"]
_SPINNER_FRAMES: list[str] = _SPINNER["frames"]  # type: ignore[assignment]  # rich Spinner dict has untyped values
_SPINNER_INTERVAL: float = _SPINNER["interval"] / 1000  # type: ignore[operator]  # rich Spinner dict has untyped values

_OUTPUT_LEVEL_STYLES: dict[OutputLevel, str] = {
    OutputLevel.INFO: STYLE_DIM,
    OutputLevel.WARNING: STYLE_WARNING,
    OutputLevel.ERROR: STYLE_ERROR,
    OutputLevel.SHELL_STDERR: STYLE_SHELL_STDERR,
}


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
    """Return the bottom `max_lines` of *renderable* for live viewporting."""
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
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════════
# UI — owns the terminal, consumes Events, renders
# ═══════════════════════════════════════════════════════════════════════


class UI:
    """Owns the Rich Live display and input handling. Consumes engine events."""

    _HISTORY_STYLES: ClassVar[dict[str, str]] = {
        "input": BG_INPUT,
        "active": BG_ACTIVE,
        "response": "",
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
        snapshot_ref: str | None = None,
        continue_session: bool = False,
    ) -> None:
        self.console = console
        self.state = state
        self.inp = InputState(history_provider=engine.store.search_history)
        self._events = events
        self._engine = engine
        self._pr_number = pr_number
        self._snapshot_ref = snapshot_ref
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
        self._error_detail: str = ""  # full diagnostic for error expand
        self._pending_tool_name: str = ""  # tool name from last ToolCallStarted
        self._pending_tool_args: str = ""  # tool args from last ToolCallStarted
        self._current_task_type: str = ""  # set from TaskStarted.task_type
        # Last finished panel — stays in Live until finalized by next input.
        self._pending_lines: list[RenderableType] | None = None
        self._pending_variant: Literal["response", "succeeded", "failed", "toolcall"] = "succeeded"
        self._pending_commands: list[str] = []
        # Startup commands (e.g. /review <n>, /session resume-last) —
        # dispatched after setup, not echoed or persisted to history.
        self._startup_commands: list[str] = []
        self._reload_after_startup: bool = False
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
                        (f"/connect {p.value}", prov.LABEL)
                        for p, prov in PROVIDERS.items()
                        if p.value.startswith(arg)
                    ]
                    if "endpoint".startswith(arg):
                        matches.append(("/connect endpoint", "OpenAI-compatible endpoint"))
                    matches.extend(
                        (f"/connect {s.key}", s.description)
                        for s in Service
                        if s.key.startswith(arg)
                    )
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
                        ("prune", "Remove stale entries"),
                        ("model", "Set embedding model"),
                        ("search", "Search the index"),
                        ("search-diag", "Search with diagnostics"),
                    ]
                    matches = [
                        (f"/index {name}", desc) for name, desc in subs if name.startswith(arg)
                    ]
                    self.inp.apply_completions(matches)
                case Command.MEMORY:
                    subs = [
                        ("list", "Active facts"),
                        ("all", "All facts (include superseded)"),
                        ("extract", "Extract facts from session"),
                        ("purge", "Delete old facts by age"),
                    ]
                    matches = [
                        (f"/memory {name}", desc) for name, desc in subs if name.startswith(arg)
                    ]
                    self.inp.apply_completions(matches)
                case Command.COMPACT:
                    if "reset".startswith(arg):
                        self.inp.apply_completions(
                            [
                                ("/compact reset", "Undo last compaction"),
                            ]
                        )
                case Command.SKILL:
                    self._complete_skill(arg)
                case Command.STATS:
                    self._complete_stats(arg)
                case Command.SESSION:
                    self._complete_session(arg)
            return

        # Complete the command name
        matches = [(c.slash, c.description) for c in Command if c.slash.startswith(text)]
        self.inp.apply_completions(matches)

    def _complete_model(self, partial: str) -> None:
        """Complete a model ID argument for /model.

        Uses the cached model list from `state.cached_models`.
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

        Uses cached PR/branch data from the last `/review` list.
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
        """Complete /session subcommands and session IDs/labels for resume/delete."""
        sub_parts = arg.split(None, 1)
        subcmd = sub_parts[0] if sub_parts else arg

        # Second-level: complete after "resume" or "delete".
        if subcmd in ("resume", "delete") and (
            len(sub_parts) == 2 or (len(sub_parts) == 1 and arg.endswith(" "))
        ):
            partial = sub_parts[1] if len(sub_parts) == 2 else ""
            sessions = self._engine.store.list_sessions(limit=50)
            current = self._engine.state.session_id
            lower = partial.lower()
            matches: list[tuple[str, str]] = []
            for s in sessions:
                if s.session_id == current:
                    continue
                short_id = s.session_id[:12]
                label = s.session_label or ""
                if subcmd == "delete":
                    # Delete requires exact ID prefix.
                    if short_id.startswith(partial):
                        matches.append((f"/session delete {short_id}", label or short_id[:8]))
                elif short_id.startswith(partial):
                    # ID prefix match — complete with the ID.
                    matches.append((f"/session resume {short_id}", label or short_id[:8]))
                elif label and lower in label.lower():
                    # Label substring match — complete with the label
                    # so common-prefix extension stays in label space.
                    matches.append((f"/session resume {label}", short_id))
            self.inp.apply_completions(matches)
            return

        # First-level: complete subcommand name.
        subs = [
            ("all", "Sessions across all repos"),
            ("history", "Last 10 inputs in this session"),
            ("info", "Current session details"),
            ("rename", "Rename the current session"),
            ("resume", "Resume a previous session"),
            ("delete", "Delete a session by ID"),
            ("purge", "Delete sessions older than duration"),
        ]
        matches = [(f"/session {name}", desc) for name, desc in subs if name.startswith(arg)]
        self.inp.apply_completions(matches)

    def _complete_skill(self, partial: str) -> None:
        """Complete a skill name argument for `/skill`."""
        registry = self._engine.state.skill_registry
        if registry is None:
            return
        matches = [
            (f"/skill {s.name}", s.description)
            for s in registry.all()
            if s.name.startswith(partial)
        ]
        self.inp.apply_completions(matches)

    def _complete_stats(self, arg: str) -> None:
        """Complete /stats arguments: `all` or session ID/label."""
        matches: list[tuple[str, str]] = []
        if "all".startswith(arg):
            matches.append(("/stats all", "Stats across all sessions"))
        lower = arg.lower()
        sessions = self._engine.store.list_sessions(limit=50)
        for s in sessions:
            short_id = s.session_id[:12]
            label = s.session_label or ""
            if short_id.startswith(arg):
                matches.append((f"/stats {short_id}", label or short_id[:8]))
            elif label and lower in label.lower():
                matches.append((f"/stats {label}", short_id))
        self.inp.apply_completions(matches)

    def _complete_shell(self) -> None:
        """Complete a shell command in a background thread.

        Delegates to `query_shell_completions` (bash completion →
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
            case TaskStarted(task_type=task_type):
                self._expandable = False
                self._active_lines.clear()
                self._active_had_error = False
                self._active_task = True
                self._streaming_text = ""
                self._streaming_md = None
                self._error_detail = ""
                self._current_task_type = task_type
            case Output(text=text, level=level, detail=detail):
                if level == OutputLevel.ERROR:
                    self._active_had_error = True
                    t = Text()
                    t.append("Error: ", style=ERROR)
                    t.append(text)
                    self._active_lines.append(t)
                    if detail:
                        self._error_detail = detail
                else:
                    self._active_lines.append(Text(text, style=_OUTPUT_LEVEL_STYLES[level]))
            case TableOutput() as te:
                table = Table(title=te.title, show_lines=False, style=STYLE_DIM)
                for col in te.columns:
                    table.add_column(
                        col.header,
                        width=col.width if col.width is not None else None,
                        style=COLUMN_BRANCH if col.highlight else "",
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
            case LinkOutput(url=url, label=label):
                if label:
                    markup = f"{label}: [link={url}][{LINK_STYLE}]{url}[/{LINK_STYLE}][/link]"
                else:
                    markup = f"[link={url}][{LINK_STYLE}]{url}[/{LINK_STYLE}][/link]"
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
                # Flush any preceding LLM text as its own panel.
                if self._active_lines:
                    self._finalize_pending()
                    preamble = Group(*self._active_lines)
                    panel = self._history_panel("response", preamble)
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
                    self._print_to_scrollback(panel)
                    self._active_lines = []
                    self._active_had_error = False
                self._streaming_text = ""
                self._streaming_md = None
                self._pending_tool_name = name
                self._pending_tool_args = args
            case ToolCallOutput() as tco:
                # Live streaming display for long-running tools.
                args = self._pending_tool_args
                args_short = args[:80] + "…" if len(args) > 80 else args
                header_text = f"⚙ {self._pending_tool_name}({args_short})"
                lines: list[RenderableType] = [Text(header_text, style=MUTED)]
                if tco.total_lines <= tco.head_lines + tco.tail_lines:
                    # Few lines — show everything.
                    combined = tco.head
                    if tco.tail:
                        combined = f"{tco.head}\n{tco.tail}" if tco.head else tco.tail
                    if combined:
                        lines.append(Text(combined, style=STYLE_DIM))
                else:
                    # Head / spacer / tail.
                    if tco.head:
                        lines.append(Text(tco.head, style=STYLE_DIM))
                    hidden = tco.total_lines - tco.head_lines - tco.tail_lines
                    spacer = f"  ⋯ {hidden} more lines ({tco.elapsed:.1f}s)"
                    lines.append(Text(spacer, style=STYLE_DIM_ITALIC))
                    if tco.tail:
                        lines.append(Text(tco.tail, style=STYLE_DIM))
                self._active_lines = lines
                if self._live:
                    self._live.update(self._render_view(), refresh=True)
            case ToolCallFinished(tool_name=_name, result=result, error=error):
                # Finalize any previous pending panel (e.g. prior tool call).
                self._finalize_pending()
                self._active_lines = []
                self._streaming_text = ""
                self._streaming_md = None
                args = self._pending_tool_args
                args_short = args[:80] + "…" if len(args) > 80 else args
                failed = error is not None
                icon = "✗" if failed else "⚙"
                body = error or result
                body_style = STYLE_ERROR if failed else STYLE_DIM
                header_text = f"{icon} {self._pending_tool_name}({args_short})"
                tool_lines: list[RenderableType] = [Text(header_text, style=MUTED)]
                hidden = 0
                if body:
                    body_split = body.splitlines()
                    max_lines = config.tui.tool_max_lines
                    if len(body_split) > max_lines:
                        shown = "\n".join(body_split[:max_lines])
                        hidden = len(body_split) - max_lines
                        tool_lines.append(Text(shown, style=body_style))
                    else:
                        tool_lines.append(Text(body, style=body_style))
                self._pending_lines = tool_lines
                self._pending_variant = "failed" if failed else "toolcall"
                if hidden:
                    self._expandable = True
                    self._expand_hidden = hidden
                    self._expand_kind = _ExpandKind.TOOL
                    self._tool_full_result = body
                    self._tool_header = header_text
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
                line = Text(f"Compacting {old} messages …", style=STYLE_DIM)
                panel = self._history_panel("queued", line)
                self._print_to_scrollback(panel)
            case CompactionFinished(summary_tokens=tokens):
                old = self._compaction_old
                if tokens > 0:
                    line = Text(
                        f"Compacted {old} messages into ~{_format_count(tokens)} tokens.",
                        style=STYLE_DIM,
                    )
                else:
                    line = Text("Compaction failed.", style=STYLE_DIM)
                panel = self._history_panel("queued", line)
                if self._live:
                    self._live.update(self._render_view(), refresh=True)
                self._print_to_scrollback(panel)
            case FactExtractionStarted():
                line = Text("Extracting facts \u2026", style=STYLE_DIM)
                panel = self._history_panel("queued", line)
                self._print_to_scrollback(panel)
            case FactExtractionFinished(added=added, confirmed=confirmed, superseded=superseded):
                total = added + confirmed + superseded
                if total > 0:
                    parts: list[str] = []
                    if added:
                        parts.append(f"{added} new")
                    if confirmed:
                        parts.append(f"{confirmed} confirmed")
                    if superseded:
                        parts.append(f"{superseded} superseded")
                    line = Text(f"Memory: {', '.join(parts)}.", style=STYLE_DIM)
                else:
                    line = Text("No new facts extracted.", style=STYLE_DIM)
                panel = self._history_panel("queued", line)
                if self._live:
                    self._live.update(self._render_view(), refresh=True)
                self._print_to_scrollback(panel)
            case ReviewPosted():
                pass  # Visual feedback handled by LinkOutput
            case ContextMarkerReady(marker=marker, content=content):
                self._insert_context_marker(marker, content)
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
                # LLM tasks with streaming markdown get transparent
                # background; all other tasks get succeeded/failed.
                is_llm_response = (
                    self._current_task_type == "llm"
                    and self._streaming_text
                    and not self._active_had_error
                )
                variant: Literal["response", "succeeded", "failed"] = (
                    "response"
                    if is_llm_response
                    else ("failed" if self._active_had_error else "succeeded")
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
                    elif self._error_detail:
                        detail_lines = self._error_detail.count("\n") + 1
                        self._expandable = True
                        self._expand_hidden = detail_lines
                        self._expand_kind = _ExpandKind.ERROR
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

        # Build a sorted list of marker spans for styled rendering.
        spans = self.inp.marker_spans()
        spans.sort(key=lambda s: s[0])

        # Walk through the buffer, emitting styled segments.
        i = 0
        span_idx = 0
        while i < len(buf):
            # Check if we're at a marker start.
            if span_idx < len(spans) and i == spans[span_idx][0]:
                m_start, m_end, region = spans[span_idx]
                marker_text = buf[m_start:m_end]
                marker_style = CONTEXT_MARKER if region.kind is MarkerKind.CONTEXT else PASTE_MARKER
                if pos == m_start:
                    # Cursor is at the marker — highlight the leading '['.
                    t.append(marker_text[0], style=CURSOR)
                    t.append(marker_text[1:], style=marker_style)
                else:
                    t.append(marker_text, style=marker_style)
                i = m_end
                span_idx += 1
                continue

            # Regular character — apply cursor highlight if needed.
            if i == pos:
                t.append(buf[i], style=CURSOR)
            else:
                t.append(buf[i])
            i += 1

        # If cursor is at end-of-buffer, show the block cursor.
        if pos >= len(buf):
            t.append(" ", style=CURSOR)

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
        return render_footer(
            self.state,
            self.console.width,
            index_ready=self._index_ready,
            index_chunks=self._index_chunks,
            index_phase=self._index_phase,
            index_indexed=self._index_indexed,
            index_total=self._index_total,
            spinner_frame=self._spinner_frame(),
        )

    def _history_panel(
        self,
        variant: Literal[
            "input", "active", "response", "succeeded", "failed", "queued", "toolcall"
        ],
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

    def _insert_context_marker(self, marker: str, content: str) -> None:
        """Insert a context marker at the start of the input buffer.

        Appends after any existing context markers so markers read
        left-to-right in execution order.  Shifts the cursor right
        so in-progress typing is undisturbed.
        """
        region = PasteRegion(marker=marker, content=content, kind=MarkerKind.CONTEXT)
        self.inp.paste_regions.append(region)

        # Find insertion point: after the last context marker.
        insert_pos = 0
        for _span_start, span_end, span_region in self.inp.marker_spans():
            if span_region.kind is MarkerKind.CONTEXT:
                insert_pos = max(insert_pos, span_end)

        text = self.inp.text
        # Add a trailing space so markers don't merge visually.
        new_text = text[:insert_pos] + marker + " " + text[insert_pos:]
        old_cursor = self.inp.cursor
        new_cursor = old_cursor + len(marker) + 1 if old_cursor >= insert_pos else old_cursor
        self.inp.set_text(new_text, cursor=new_cursor)

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
            case _ExpandKind.ERROR:
                self._expand_error()

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
        # Determine style based on current pending variant (failed = error style).
        text_style = STYLE_ERROR if self._pending_variant == "failed" else STYLE_DIM
        self._pending_lines = [
            Text(self._tool_header, style=MUTED),
            Text(self._tool_full_result, style=text_style),
        ]
        self._tool_full_result = ""
        self._tool_header = ""
        self._finalize_pending()

    def _expand_error(self) -> None:
        """Expand an error panel with full diagnostic detail."""
        if not self._error_detail:
            return
        self._expandable = False
        self._expand_kind = None
        lines: list[RenderableType] = list(self._pending_lines or [])
        lines.append(Text(self._error_detail, style=STYLE_DIM))
        self._pending_lines = lines
        self._error_detail = ""
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
            elif self._snapshot_ref is not None:
                self._startup_commands.append(f"/review {self._snapshot_ref}")
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
                # Reload history after the last one finishes so
                # resumed session commands appear in up-arrow.
                if not self._active_task and self._startup_commands:
                    cmd = self._startup_commands.pop(0)
                    self._start_task(TaskType.COMMAND, cmd, persist=False)
                    if not self._startup_commands:
                        self._reload_after_startup = True
                    live.update(self._render_view())
                if self._reload_after_startup and not self._active_task:
                    self._reload_after_startup = False
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
    snapshot_ref: str | None = None,
    continue_session: bool = False,
) -> None:
    """Launch the rbtr interactive session."""
    theme = build_theme(config.theme)
    console = Console(markup=True, highlight=False, theme=theme)
    state = EngineState()
    events: queue.Queue[Event] = queue.Queue()
    with Engine(state, events) as engine:
        ui = UI(
            console,
            state,
            events,
            engine,
            pr_number=pr_number,
            snapshot_ref=snapshot_ref,
            continue_session=continue_session,
        )
        ui.run()
