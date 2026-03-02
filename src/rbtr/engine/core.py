"""Engine — runs commands in daemon threads, emits Events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import signal
import subprocess
import sys
import threading

from rbtr.events import Event, FlushPanel, MarkdownOutput, Output, TaskFinished, TaskStarted
from rbtr.exceptions import RbtrError, TaskCancelled
from rbtr.llm import LLMContext, compact_history, handle_llm, reset_compaction
from rbtr.models import BranchTarget, PRTarget, Target
from rbtr.sessions.store import SESSIONS_DB_PATH, SessionStore
from rbtr.state import EngineState
from rbtr.styles import STYLE_DIM, STYLE_ERROR, STYLE_WARNING

from .connect_cmd import cmd_connect
from .draft_cmd import cmd_draft
from .index_cmd import cmd_index
from .model_cmd import cmd_model
from .review_cmd import cmd_review
from .session_cmd import cmd_session
from .setup import run_setup
from .shell import handle_shell
from .stats_cmd import cmd_stats
from .types import Command, TaskType

log = logging.getLogger(__name__)


class Engine:
    """Executes commands and emits typed events. Knows nothing about the UI."""

    def __init__(
        self,
        state: EngineState,
        events: queue.Queue[Event],
        *,
        store: SessionStore | None = None,
    ) -> None:
        self.state = state
        self.events = events
        self._last_shell_full_output: tuple[str, str, int, int] | None = (
            None  # (stdout, stderr, returncode, hidden_count)
        )
        self._cancel = threading.Event()
        self._shell_proc: subprocess.Popen[str] | None = None
        self._shell_pgid: int | None = None
        # Session persistence store.
        self.store = store if store is not None else SessionStore(SESSIONS_DB_PATH)
        state.session_id = self.store.new_id()
        # Long-lived event loop for async work (LLM streaming).
        # Keeps httpx connection pools alive across calls.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="asyncio-loop"
        )
        self._loop_thread.start()

    # ── Cancellation ─────────────────────────────────────────────────

    def cancel(self) -> None:
        """Signal the running task to stop. Thread-safe."""
        self._cancel.set()
        pgid = self._shell_pgid
        if pgid is not None:
            with contextlib.suppress(OSError):
                os.killpg(pgid, signal.SIGTERM)

    def _check_cancel(self) -> None:
        """Raise TaskCancelled if cancellation was requested."""
        if self._cancel.is_set():
            raise TaskCancelled

    # ── Event emitters ───────────────────────────────────────────────

    def _emit(self, event: Event) -> None:
        self.events.put(event)

    def _out(self, text: str, style: str = STYLE_DIM) -> None:
        self._check_cancel()
        self._emit(Output(text=text, style=style))

    def _warn(self, text: str) -> None:
        self._out(text, style=STYLE_WARNING)

    def _error(self, text: str) -> None:
        self._out(text, style=STYLE_ERROR)

    def _markdown(self, text: str) -> None:
        self._check_cancel()
        self._emit(MarkdownOutput(text=text))

    def _flush(self) -> None:
        """Flush current output as a completed panel and start fresh."""
        self._emit(FlushPanel())

    def _clear(self) -> None:
        """Discard current transient output (e.g. "Fetching…" messages)."""
        self._emit(FlushPanel(discard=True))

    def _sync_store_context(self) -> None:
        """Push current engine metadata to the store context.

        Called before each task so ``save_messages`` / ``save_input`` /
        ``compact_session`` inherit metadata without explicit kwargs.
        """
        self.store.set_context(
            self.state.session_id,
            session_label=self.state.session_label,
            repo_owner=self.state.owner,
            repo_name=self.state.repo_name,
            model_name=self.state.model_name,
            review_target=_review_target_str(self.state.review_target),
        )

    # ── LLM context ───────────────────────────────────────────────────

    def _llm_context(self) -> LLMContext:
        """Build an :class:`LLMContext` for the LLM pipeline."""
        return LLMContext(
            state=self.state,
            store=self.store,
            events=self.events,
            cancel=self._cancel,
            loop=self._loop,
        )

    # ── Task runner ──────────────────────────────────────────────────

    def run_task(self, task_type: TaskType, arg: str, *, persist: bool = True) -> None:
        """Run a task synchronously (called from a daemon thread)."""
        self._emit(TaskStarted(task_id=f"{task_type}:{arg}"))
        self._sync_store_context()
        success = True
        cancelled = False
        try:
            match task_type:
                case TaskType.SETUP:
                    run_setup(self)
                case TaskType.COMMAND:
                    if persist:
                        self._persist_input(arg, "command")
                    self._handle_command(arg)
                case TaskType.SHELL:
                    self._persist_input(f"!{arg}", "shell")
                    handle_shell(self, arg)
                case TaskType.LLM:
                    handle_llm(self._llm_context(), arg)
        except TaskCancelled:
            success = False
            cancelled = True
        except RbtrError as e:
            self._error(str(e))
            success = False
        except Exception as e:
            self._error(f"Unexpected error: {e}")
            success = False
        self._emit(TaskFinished(success=success, cancelled=cancelled))

    # ── Commands ─────────────────────────────────────────────────────

    def _handle_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        args = parts[1].strip() if len(parts) > 1 else ""

        try:
            cmd = Command(parts[0].lower())
        except ValueError:
            self._warn(f"Unknown command: {parts[0]}. Type /help for commands.")
            return

        match cmd:
            case Command.HELP:
                self._cmd_help()
            case Command.REVIEW:
                cmd_review(self, args)
            case Command.DRAFT:
                cmd_draft(self, args)
            case Command.CONNECT:
                cmd_connect(self, args)
            case Command.MODEL:
                cmd_model(self, args)
            case Command.INDEX:
                cmd_index(self, args)
            case Command.COMPACT:
                ctx = self._llm_context()
                sub = args.split(maxsplit=1)[0].lower() if args else ""
                if sub == "reset":
                    reset_compaction(ctx)
                else:
                    compact_history(ctx, extra_instructions=args)
            case Command.SESSION:
                cmd_session(self, args)
            case Command.STATS:
                cmd_stats(self, args)
            case Command.NEW:
                self._cmd_new()
            case Command.QUIT:
                pass  # handled by caller

    def _cmd_help(self) -> None:
        for c in Command:
            self._emit(
                Output(
                    text=f"  {c.slash:<12}{c.description}",
                    style=STYLE_DIM,
                )
            )

    def _cmd_new(self) -> None:
        import time

        self.state.usage.reset()
        self.state.session_id = self.store.new_id()
        self.state.session_started_at = time.time()
        self._out("Conversation cleared.")

    def _persist_input(self, text: str, kind: str) -> None:
        """Save a command or shell input to the session store."""
        try:
            self.store.save_input(self.state.session_id, text, kind)
        except OSError:
            log.warning("sessions: failed to persist input", exc_info=True)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Release resources — call on shutdown."""
        self.store.close()

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _copy_to_clipboard(text: str) -> None:
        if sys.platform == "darwin":
            with contextlib.suppress(FileNotFoundError, subprocess.SubprocessError):
                subprocess.run(
                    ["pbcopy"],  # noqa: S607
                    input=text.encode("utf-8"),
                    check=True,
                    timeout=2,
                )


def _review_target_str(target: Target | None) -> str | None:
    """Encode a review target as the string you'd pass to ``/review``.

    ``"42"`` for PRs, ``"main feature"`` for branches.
    """
    match target:
        case PRTarget(number=n):
            return str(n)
        case BranchTarget(base_branch=base, head_branch=head):
            return f"{base} {head}"
        case _:
            return None
