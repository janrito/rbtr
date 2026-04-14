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

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.segment import Segment
from rich.spinner import SPINNERS
from rich.table import Table
from rich.text import Text

from rbtr_legacy.config import ThinkingEffort, config
from rbtr_legacy.engine.types import Command, Service, TaskType
from rbtr_legacy.events import (
    AutoSubmit,
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
    InputEcho,
    LinkOutput,
    MarkdownOutput,
    Output,
    OutputLevel,
    PanelVariant,
    ReviewPosted,
    TableOutput,
    TaskFinished,
    TaskStarted,
    TextDelta,
    ToolCallFinished,
    ToolCallOutput,
    ToolCallStarted,
)
from rbtr_legacy.state import EngineState
from rbtr_legacy.styles import (
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
from rbtr_legacy.tui.footer import _format_count, render_footer
from rbtr_legacy.tui.input import (
    InputReader,
    InputState,
    query_shell_completions,
)

if TYPE_CHECKING:
    from rbtr_legacy.engine.core import Engine

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


# ── Head/tail truncation ─────────────────────────────────────────────


def _truncate_head_tail(
    text: str,
    head_max: int,
    tail_max: int,
) -> tuple[str, int, str]:
    """Split *text* into head, hidden count, and tail.

    Returns `(head, hidden, tail)`.  When the text fits within
    `head_max + tail_max` lines, `hidden` is 0 and `tail` is
    empty (all content is in `head`).
    """
    lines = text.splitlines()
    total = len(lines)
    budget = head_max + tail_max
    if total <= budget:
        return text, 0, ""
    head = "\n".join(lines[:head_max])
    tail = "\n".join(lines[-tail_max:])
    return head, total - budget, tail


def _render_head_tail(
    head: str,
    hidden: int,
    tail: str,
    style: str,
    *,
    elapsed: float | None = None,
) -> list[RenderableType]:
    """Build Rich renderables for a head/ellipsis/tail display.

    When `hidden` is 0, returns the full text as a single `Text`.
    When `elapsed` is set, the spacer includes the elapsed time
    (used for streaming `ToolCallOutput`).
    """
    parts: list[RenderableType] = []
    if head:
        parts.append(Text(head, style=style))
    if not hidden:
        return parts
    if elapsed is not None:
        spacer = f"  ⋯ {hidden} more lines ({elapsed:.1f}s)"
    else:
        spacer = f"  ⋯ {hidden} more lines"
    parts.append(Text(spacer, style=STYLE_DIM_ITALIC))
    if tail:
        parts.append(Text(tail, style=style))
    return parts


@dataclass
class _LivePanel:
    """A panel in the Live area awaiting finalization to scrollback.

    Used for concurrent tool calls (N panels), shell output, and
    error details.  Holds both the truncated and optionally the
    expanded renderable lines so ctrl+O can swap them without
    re-reading external state.
    """

    lines: list[RenderableType]
    variant: PanelVariant
    done: bool = False
    hidden: int = 0
    expanded_lines: list[RenderableType] | None = None
    # Tool-call panels only — used by ToolCallOutput to rebuild
    # the header during streaming.  Empty for non-tool panels.
    tool_name: str = ""
    tool_args: str = ""


# ═══════════════════════════════════════════════════════════════════════
# UI — owns the terminal, consumes Events, renders
# ═══════════════════════════════════════════════════════════════════════


class UI:
    """Owns the Rich Live display and input handling. Consumes engine events."""

    _HISTORY_STYLES: ClassVar[dict[PanelVariant, str]] = {
        PanelVariant.INPUT: BG_INPUT,
        PanelVariant.ACTIVE: BG_ACTIVE,
        PanelVariant.RESPONSE: "",
        PanelVariant.SUCCEEDED: BG_SUCCEEDED,
        PanelVariant.FAILED: BG_FAILED,
        PanelVariant.QUEUED: BG_QUEUED,
        PanelVariant.TOOLCALL: BG_TOOLCALL,
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
        self._error_detail: str = ""  # accumulated during task, used at TaskFinished
        self._live_panels: dict[str, _LivePanel] = {}

        self._current_task_type: str = ""  # set from TaskStarted.task_type
        self._pending_commands: list[str] = []
        # Messages queued by engine commands for auto-submission to the LLM.
        # Dispatched after the command task finishes.
        self._auto_submit: list[AutoSubmit] = []
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
                    from rbtr_legacy.providers import (
                        PROVIDERS,  # deferred: avoids pydantic_ai at import time
                    )

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
                from rbtr_legacy.engine.model_cmd import (
                    get_models,  # deferred: avoids loading core.py at import time
                )

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
                from rbtr_legacy.git import list_local_branches

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
        from rbtr_legacy.engine.draft_cmd import POST_EVENTS, SUBCOMMANDS

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
        # Finalize a completed tool batch when a non-tool event arrives.
        # Exception: TaskFinished handles the batch itself (keeps it
        # live for ctrl+O when there's no text after the tools).
        if (
            self._live_panels
            and self._all_panels_done()
            and not isinstance(
                event, (ToolCallStarted, ToolCallOutput, ToolCallFinished, TaskFinished, InputEcho)
            )
        ):
            self._finalize_panels()

        match event:
            case InputEcho(text=text):
                self._echo_input(text)
            case TaskStarted(task_type=task_type):
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
            case FlushPanel(discard=discard, variant=declared_variant):
                if not discard and self._active_lines:
                    flush_variant = declared_variant or (
                        PanelVariant.FAILED if self._active_had_error else PanelVariant.SUCCEEDED
                    )
                    content: RenderableType = Group(*self._active_lines)
                    panel = self._history_panel(flush_variant, content)
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
                    self._print_to_scrollback(panel)
                self._active_lines = []
                self._active_had_error = False
            case ToolCallStarted(tool_name=name, args=args, tool_call_id=cid):
                # Finalize a completed batch from a previous node.
                if self._live_panels and self._all_panels_done():
                    self._finalize_panels()
                # Flush any preceding LLM text as its own panel.
                if self._active_lines:
                    preamble = Group(*self._active_lines)
                    panel = self._history_panel(PanelVariant.RESPONSE, preamble)
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
                    self._print_to_scrollback(panel)
                    self._active_lines = []
                    self._active_had_error = False
                self._streaming_text = ""
                self._streaming_md = None
                header = self._format_tool_header("⚙", name, args)
                entry = _LivePanel(
                    lines=[
                        Text(header, style=MUTED),
                        Text("  running…", style=STYLE_DIM_ITALIC),
                    ],
                    variant=PanelVariant.TOOLCALL,
                    tool_name=name,
                    tool_args=args,
                )
                self._live_panels[cid] = entry
                if self._live:
                    self._live.update(self._render_view(), refresh=True)
            case ToolCallOutput(tool_call_id=cid) as tco:
                tool = self._live_panels.get(cid)
                if tool is not None:
                    header_text = self._format_tool_header("⚙", tool.tool_name, tool.tool_args)
                    hidden = tco.total_lines - tco.head_lines - tco.tail_lines
                    tool.lines = [Text(header_text, style=MUTED)]
                    tool.lines.extend(
                        _render_head_tail(
                            tco.head, hidden, tco.tail, STYLE_DIM, elapsed=tco.elapsed
                        )
                    )
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
            case ToolCallFinished(tool_call_id=cid, result=result, error=error):
                tool = self._live_panels.get(cid)
                if tool is not None:
                    failed = error is not None
                    icon = "✗" if failed else "⚙"
                    body = error or result
                    body_style = STYLE_ERROR if failed else STYLE_DIM
                    header_text = self._format_tool_header(icon, tool.tool_name, tool.tool_args)
                    head, hidden, tail = _truncate_head_tail(
                        body,
                        config.tui.tool_head_lines,
                        config.tui.tool_tail_lines,
                    )
                    tool.lines = [Text(header_text, style=MUTED)]
                    tool.lines.extend(_render_head_tail(head, hidden, tail, body_style))
                    tool.done = True
                    tool.variant = PanelVariant.FAILED if failed else PanelVariant.TOOLCALL
                    tool.hidden = hidden
                    if hidden:
                        tool.expanded_lines = [
                            Text(header_text, style=MUTED),
                            Text(body, style=body_style),
                        ]
                    if self._live:
                        self._live.update(self._render_view(), refresh=True)
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
                panel = self._history_panel(PanelVariant.QUEUED, line)
                self._print_to_scrollback(panel)
            case CompactionFinished(summary_tokens=tokens, summary_preview=preview):
                old = self._compaction_old
                if tokens > 0:
                    header = f"Compacted {old} messages into ~{_format_count(tokens)} tokens."
                    line = Text(header, style=STYLE_DIM)
                    if preview:
                        line.append("\n")
                        line.append(preview, style=STYLE_DIM)
                else:
                    line = Text("Compaction failed.", style=STYLE_DIM)
                panel = self._history_panel(PanelVariant.QUEUED, line)
                if self._live:
                    self._live.update(self._render_view(), refresh=True)
                self._print_to_scrollback(panel)
            case FactExtractionStarted():
                line = Text("Extracting facts \u2026", style=STYLE_DIM)
                panel = self._history_panel(PanelVariant.QUEUED, line)
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
                panel = self._history_panel(PanelVariant.QUEUED, line)
                if self._live:
                    self._live.update(self._render_view(), refresh=True)
                self._print_to_scrollback(panel)
            case ReviewPosted():
                pass  # Visual feedback handled by LinkOutput
            case ContextMarkerReady(marker=marker, content=content):
                self.inp.add_context(marker, content)
            case AutoSubmit() as auto:
                self._auto_submit.append(auto)
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
                    self._auto_submit.clear()
                self._active_task = False
                # LLM tasks with streaming markdown get transparent
                # background; all other tasks get succeeded/failed.
                is_llm_response = (
                    self._current_task_type == "llm"
                    and self._streaming_text
                    and not self._active_had_error
                )
                is_setup = self._current_task_type == "setup"
                variant = (
                    PanelVariant.RESPONSE
                    if is_llm_response or is_setup
                    else (PanelVariant.FAILED if self._active_had_error else PanelVariant.SUCCEEDED)
                )
                # Handle in-flight tool batch.
                if self._live_panels:
                    if cancelled:
                        # Mark pending tools as cancelled.
                        for tool in self._live_panels.values():
                            if not tool.done:
                                header = self._format_tool_header(
                                    "✗", tool.tool_name, tool.tool_args
                                )
                                tool.lines = [
                                    Text(header, style=MUTED),
                                    Text("  Cancelled.", style=STYLE_WARNING),
                                ]
                                tool.done = True
                                tool.variant = PanelVariant.FAILED
                    if self._active_lines:
                        # Text after tools — finalize batch, then
                        # handle the text below.
                        self._finalize_panels()
                    # else: keep batch live for ctrl+O.

                has_active = bool(self._active_lines)
                if has_active:
                    # There's content after the last tool call (or no
                    # tool calls at all).
                    info = self._engine._last_shell_full_output
                    if info:
                        # Shell output — build expanded lines eagerly.
                        stdout_full, stderr_full, returncode, hidden = info
                        self._engine._last_shell_full_output = None
                        echo = self._active_lines[0] if self._active_lines else None
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
                        self._live_panels["_task"] = _LivePanel(
                            lines=self._active_lines,
                            variant=variant,
                            done=True,
                            hidden=hidden,
                            expanded_lines=expanded,
                        )
                    elif self._error_detail:
                        # Error detail — build expanded lines eagerly.
                        detail_lines = self._error_detail.count("\n") + 1
                        expanded_err = list(self._active_lines)
                        expanded_err.append(Text(self._error_detail, style=STYLE_DIM))
                        self._error_detail = ""
                        self._live_panels["_task"] = _LivePanel(
                            lines=self._active_lines,
                            variant=variant,
                            done=True,
                            hidden=detail_lines,
                            expanded_lines=expanded_err,
                        )
                    else:
                        content = Group(*self._active_lines) if self._active_lines else Text("")
                        panel = self._history_panel(variant, content)
                        if self._live:
                            self._live.update(self._render_view(), refresh=True)
                        self._print_to_scrollback(panel)
                # else: no active content — keep live panels
                # as-is so the user can Ctrl+O to expand.
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

    def _render_context_line(self) -> Text | None:
        """Render context markers as a line above the prompt.

        Returns `None` when there are no context regions.
        """
        if not self.inp.context_regions:
            return None
        t = Text()
        for i, region in enumerate(self.inp.context_regions):
            if i > 0:
                t.append(" ")
            t.append(region.marker, style=CONTEXT_MARKER)
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
                m_start, m_end, _region = spans[span_idx]
                marker_text = buf[m_start:m_end]
                marker_style = PASTE_MARKER
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
        if self._live_panels:
            # Live panels — tool calls, shell output, errors.
            any_expandable = self._all_panels_done() and any(
                p.expanded_lines is not None for p in self._live_panels.values()
            )
            for panel in self._live_panels.values():
                top_parts.append(Text(""))
                lines = list(panel.lines)
                bottom = 1
                if panel.done and panel.hidden:
                    bottom = 0
                top_parts.append(
                    self._history_panel(panel.variant, Group(*lines), bottom_pad=bottom)
                )
                if panel.done and panel.hidden:
                    bg = self._HISTORY_STYLES[panel.variant]
                    hint_text = f"  … {panel.hidden} more lines"
                    if any_expandable:
                        hint_text += " (ctrl+o to expand)"
                    top_parts.append(
                        Padding(Text(hint_text, style=STYLE_DIM_ITALIC), (0, 2, 1, 2), style=bg)
                    )
            # Also show active lines (e.g. streaming text after tools).
            if self._active_lines:
                top_parts.append(Text(""))
                active = Group(*self._active_lines)
                top_parts.append(self._history_panel(PanelVariant.ACTIVE, active))
        elif self._active_task:
            # Margin above active panel — matches the margin that
            # _print_to_scrollback adds for finalized panels.
            top_parts.append(Text(""))
            content = Group(*self._active_lines) if self._active_lines else Text("")
            top_parts.append(self._history_panel(PanelVariant.ACTIVE, content))
        if self._pending_commands:
            lines = [Text(f"  > {cmd}", style=MUTED) for cmd in self._pending_commands]
            top_parts.append(Text(""))
            top_parts.append(self._history_panel(PanelVariant.QUEUED, Group(*lines)))

        chrome_parts: list[RenderableType] = [Text("")]
        if self._active_task:
            frame = self._spinner_frame()
            chrome_parts.append(Rule(title=f"[{DIM}]{frame}[/{DIM}]", style=RULE))
        else:
            chrome_parts.append(Rule(style=RULE))
        context_line = self._render_context_line()
        if context_line is not None:
            chrome_parts.append(context_line)
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
        variant: PanelVariant,
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

    def _format_tool_header(self, icon: str, name: str, args: str) -> str:
        """Build the `⚙ name(args)` header for a tool call.

        For `run_command`, detects skill execution and shows
        `[skill · source] relative_cmd` instead of raw JSON.
        """

        # Skill-aware shortening for run_command.
        if name == "run_command":
            registry = self._engine.state.skill_registry
            if registry:
                try:
                    command = json.loads(args).get("command", "")
                except (json.JSONDecodeError, AttributeError):
                    command = ""
                if command:
                    match = registry.match_command(command)
                    if match:
                        skill, rel = match
                        return f"{icon} [{skill.name} · {skill.source}] {rel}"

        args_short = args[:80] + "…" if len(args) > 80 else args
        return f"{icon} {name}({args_short})"

    def _all_panels_done(self) -> bool:
        """True when all live panels have finished."""
        return bool(self._live_panels) and all(p.done for p in self._live_panels.values())

    def _finalize_panels(self) -> None:
        """Move all live panels to scrollback."""
        for panel in self._live_panels.values():
            lines = list(panel.lines)
            if panel.hidden:
                lines.append(Text(f"  … {panel.hidden} more lines", style=STYLE_DIM_ITALIC))
            content = Group(*lines) if lines else Text("")
            self._print_to_scrollback(self._history_panel(panel.variant, content))
        self._live_panels.clear()

    def _reload_history(self) -> None:
        """Reload up-arrow history from the DB."""
        provider = self.inp.history_provider
        if provider is not None:
            self.inp.history = list(reversed(provider(None, config.tui.max_history)))

    def _echo_input(self, text: str) -> None:
        """Print an Input HistoryPanel to native scrollback."""
        if self._live_panels:
            self._finalize_panels()
        t = Text()
        t.append("> ", style=PROMPT)
        t.append(text, style=INPUT_TEXT)
        self._print_to_scrollback(self._history_panel(PanelVariant.INPUT, t))

    def _expand_last_output(self) -> None:
        """Expand all truncated live panels, then finalize to scrollback."""
        if not self._live_panels or not self._all_panels_done():
            return
        self._expand_panels()

    def _expand_panels(self) -> None:
        """Expand all truncated panels, then finalize to scrollback."""
        for panel in self._live_panels.values():
            if panel.expanded_lines is not None:
                panel.lines = panel.expanded_lines
                panel.expanded_lines = None
                panel.hidden = 0
        self._finalize_panels()

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
                    self._auto_submit.clear()
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

                # Process auto-submitted messages from engine commands.
                # Process auto-submitted messages from engine commands.
                # No InputEcho — the command was already echoed when
                # the user submitted it.  No append_history — the
                # Enter handler already added the raw command.
                if not self._active_task and self._auto_submit:
                    submit = self._auto_submit.pop(0)
                    self._dispatch(submit.message)
                    live.update(self._render_view())

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
                    self._handle_event(InputEcho(text=raw))
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
    from rbtr_legacy.engine.core import (
        Engine,  # deferred: avoids pydantic_ai/PyGithub at import time
    )

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
