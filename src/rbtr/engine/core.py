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
from datetime import UTC, datetime, timedelta

import pygit2
from github import Auth, Github

from rbtr.config import config
from rbtr.creds import creds
from rbtr.events import Event, FlushPanel, Output, TaskFinished, TaskStarted
from rbtr.exceptions import RbtrError
from rbtr.oauth import oauth_is_set
from rbtr.providers import endpoint as endpoint_provider, model_context_window
from rbtr.sessions.store import SESSIONS_DB_PATH, SessionStore
from rbtr.styles import STYLE_DIM, STYLE_ERROR, STYLE_WARNING

from .compact import compact_history
from .connect import cmd_connect
from .draft_cmd import cmd_draft
from .index_cmd import cmd_index
from .llm import handle_llm
from .model import cmd_model, get_models
from .review import cmd_review
from .session_cmd import cmd_session
from .shell import handle_shell
from .state import EngineState
from .types import Command, TaskCancelled, TaskType

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
        # EngineState persistence store.
        self._store = store if store is not None else SessionStore(SESSIONS_DB_PATH)
        state.session_id = self._store.new_id()
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

    def _flush(self) -> None:
        """Flush current output as a completed panel and start fresh."""
        self._emit(FlushPanel())

    def _clear(self) -> None:
        """Discard current transient output (e.g. "Fetching…" messages)."""
        self._emit(FlushPanel(discard=True))

    # ── Task runner ──────────────────────────────────────────────────

    def run_task(self, task_type: TaskType, arg: str) -> None:
        """Run a task synchronously (called from a daemon thread)."""
        self._emit(TaskStarted(task_id=f"{task_type}:{arg}"))
        success = True
        cancelled = False
        try:
            match task_type:
                case TaskType.SETUP:
                    self._run_setup()
                case TaskType.COMMAND:
                    self._handle_command(arg)
                case TaskType.SHELL:
                    handle_shell(self, arg)
                case TaskType.LLM:
                    handle_llm(self, arg)
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

    # ── Setup ────────────────────────────────────────────────────────

    def _run_setup(self) -> None:
        # Deferred: open_repo calls pygit2.discover_repository which needs CWD set.
        from rbtr.git import open_repo
        from rbtr.github.client import parse_github_remote

        try:
            repo = open_repo()
            owner, repo_name = parse_github_remote(repo)
        except RbtrError as e:
            self._error(str(e))
            return

        self.state.repo = repo
        self.state.owner = owner
        self.state.repo_name = repo_name
        self.state.session_label = _make_session_label(owner, repo_name, repo)
        self._out(f"Repository: {owner}/{repo_name}")

        if creds.github_token:
            gh = Github(auth=Auth.Token(creds.github_token), timeout=config.github.timeout)
            self.state.gh = gh
            self.state.gh_username = gh.get_user().login
            self._out("Authenticated with GitHub.")
        else:
            self._out("Not authenticated. Use /connect github to authenticate.")

        if oauth_is_set(creds.claude):
            self.state.claude_connected = True
            self._out("Connected to Anthropic.")

        if oauth_is_set(creds.chatgpt):
            self.state.chatgpt_connected = True
            self._out("Connected to ChatGPT.")

        if creds.openai_api_key:
            self.state.openai_connected = True
            self._out("Connected to OpenAI.")

        endpoints = endpoint_provider.list_endpoints()
        for ep in endpoints:
            self._out(f"Endpoint: {ep.name} ({ep.base_url})")

        # Pre-populate model cache so Tab completion is instant.
        get_models(self)

        if not (self.state.has_llm or endpoints):
            self._out("No LLM connected. Use /connect claude, chatgpt, or openai.")

        # Load saved model preference
        saved_model = config.model
        if saved_model:
            self.state.model_name = saved_model
            _init_context_window(self)

        # Prune old sessions in the background — silent, best-effort.
        _prune_sessions(self)

        self._out("Type a message for the LLM, /help for commands, !cmd for shell")

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
                compact_history(self, extra_instructions=args)
            case Command.SESSION:
                cmd_session(self, args)
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
        self.state.message_history.clear()
        self.state.usage.reset()
        self.state.session_id = self._store.new_id()
        self.state.saved_count = 0
        self._out("Conversation cleared.")

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Release resources — call on shutdown."""
        self._store.close()

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


def _init_context_window(engine: Engine) -> None:
    """Set the context window from model metadata at startup.

    Called once when the saved model is loaded so the footer shows the
    correct context window immediately, not just after the first LLM
    response.  Works for both custom endpoints and built-in providers.
    """
    ctx = model_context_window(engine.state.model_name)
    if ctx is not None:
        engine.state.usage.context_window = ctx
        engine.state.usage.context_window_known = True


def _prune_sessions(engine: Engine) -> None:
    """Remove old sessions based on config limits.

    Runs silently — pruning failures are logged, not shown to the user.
    """
    try:
        age_days = config.sessions.max_age_days
        if age_days > 0:
            cutoff = datetime.now(UTC) - timedelta(days=age_days)
            engine._store.delete_old_sessions(before=cutoff)

        max_sessions = config.sessions.max_sessions
        if max_sessions > 0:
            engine._store.delete_excess_sessions(keep=max_sessions)
    except OSError:
        log.warning("sessions: pruning failed", exc_info=True)


def _make_session_label(owner: str, repo_name: str, repo: pygit2.Repository) -> str:
    """Build a human-readable session label from repo context.

    Format: ``owner/repo — ref`` where *ref* is the current branch
    name or short commit hash.
    """
    ref = ""
    if not repo.head_is_unborn:
        ref = str(repo.head.target)[:8] if repo.head_is_detached else repo.head.shorthand
    return f"{owner}/{repo_name} — {ref}" if ref else f"{owner}/{repo_name}"
