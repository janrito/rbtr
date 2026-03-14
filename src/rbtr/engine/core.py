"""Engine — runs commands in daemon threads, emits Events."""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import traceback

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal

from rbtr.config import config
from rbtr.events import (
    ContextMarkerReady,
    Event,
    FlushPanel,
    MarkdownOutput,
    Output,
    OutputLevel,
    TaskFinished,
    TaskStarted,
)
from rbtr.exceptions import RbtrError, TaskCancelled
from rbtr.llm import LLMContext, compact_history, handle_llm, reset_compaction
from rbtr.llm.context import CancelSlot
from rbtr.models import BranchTarget, PRTarget, SnapshotTarget, Target
from rbtr.sessions.scrub import scrub_secrets
from rbtr.sessions.store import SESSIONS_DB_PATH, SessionStore
from rbtr.state import EngineState

from .connect_cmd import cmd_connect
from .draft_cmd import cmd_draft
from .index_cmd import cmd_index
from .memory_cmd import cmd_memory
from .model_cmd import cmd_model
from .reload_cmd import cmd_reload
from .review_cmd import cmd_review
from .session_cmd import cmd_session
from .setup import run_setup
from .shell import handle_shell
from .stats_cmd import cmd_stats
from .types import Command, TaskType

log = logging.getLogger(__name__)


async def _capture_token() -> anyio.lowlevel.EventLoopToken:
    """Return the ``anyio`` token for the current event loop."""
    return anyio.lowlevel.current_token()


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
        self._cancel_slot: CancelSlot = [None]
        self._shell_proc: subprocess.Popen[str] | None = None
        self._shell_pgid: int | None = None
        # Session persistence store.
        self.store = store if store is not None else SessionStore(SESSIONS_DB_PATH)
        state.session_id = self.store.new_id()
        # Async portal for LLM streaming (keeps httpx pools alive).
        self._portal_cm = start_blocking_portal(backend="asyncio")
        self._portal: BlockingPortal = self._portal_cm.__enter__()
        self._anyio_token = self._portal.call(_capture_token)

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self) -> Engine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # ── Cancellation ─────────────────────────────────────────────────

    def cancel(self) -> None:
        """Signal the running task to stop. Thread-safe."""
        self._cancel.set()
        # Signal the anyio cancel watcher (zero-latency bridge).
        evt = self._cancel_slot[0]
        if evt is not None:
            with contextlib.suppress(Exception):
                anyio.from_thread.run_sync(evt.set, token=self._anyio_token)
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

    def _out(self, text: str, level: OutputLevel = OutputLevel.INFO) -> None:
        self._check_cancel()
        self._emit(Output(text=text, level=level))

    def _warn(self, text: str) -> None:
        self._out(text, level=OutputLevel.WARNING)

    def _error(self, text: str) -> None:
        self._out(text, level=OutputLevel.ERROR)

    def _markdown(self, text: str) -> None:
        self._check_cancel()
        self._emit(MarkdownOutput(text=text))

    def _context(self, marker: str, content: str) -> None:
        """Emit a context marker for the LLM conversation.

        The TUI inserts *marker* into the input buffer. On submit
        it expands to *content*.  The user can delete the marker
        to exclude the context.
        """
        self._emit(ContextMarkerReady(marker=marker, content=content))

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
            portal=self._portal,
            anyio_token=self._anyio_token,
            cancel_slot=self._cancel_slot,
        )

    # ── Task runner ──────────────────────────────────────────────────

    def run_task(self, task_type: TaskType, arg: str, *, persist: bool = True) -> None:
        """Run a task synchronously (called from a daemon thread)."""
        self._emit(TaskStarted(task_type=str(task_type)))
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
            self._emit(
                Output(
                    text=f"Unexpected error: {e}",
                    level=OutputLevel.ERROR,
                    detail=scrub_secrets("".join(traceback.format_exception(e))),
                )
            )
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
                    self._context("[/compact reset]", "Reset last compaction.")
                else:
                    compact_history(ctx, extra_instructions=args)
                    self._context(
                        "[/compact]",
                        f"Compacted conversation history (keeping {config.compaction.keep_turns} recent turns).",
                    )
            case Command.SESSION:
                cmd_session(self, args)
            case Command.STATS:
                cmd_stats(self, args)
            case Command.MEMORY:
                cmd_memory(self, args)
            case Command.RELOAD:
                cmd_reload(self)
            case Command.NEW:
                self._cmd_new()
            case Command.QUIT:
                pass  # handled by caller

    def _cmd_help(self) -> None:
        for c in Command:
            self._out(f"  {c.slash:<12}{c.description}")

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
        with contextlib.suppress(ImportError, OSError):
            from rbtr.index.embeddings import reset_model

            reset_model()
        self.store.close()
        with contextlib.suppress(Exception):
            self._portal_cm.__exit__(None, None, None)

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
    """Encode a review target as the string you'd pass to `/review`.

    `"42"` for PRs, `"main feature"` for branches,
    `"main"` for snapshots.
    """
    match target:
        case PRTarget(number=n):
            return str(n)
        case BranchTarget(base_branch=base, head_branch=head):
            return f"{base} {head}"
        case SnapshotTarget(ref_label=label):
            return label
        case None:
            return None
