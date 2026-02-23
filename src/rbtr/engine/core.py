"""Engine — runs commands in daemon threads, emits Events."""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import signal
import subprocess
import sys
import threading

from github import Auth, Github

from rbtr.config import config
from rbtr.creds import creds
from rbtr.events import Event, FlushPanel, Output, TaskFinished, TaskStarted
from rbtr.exceptions import RbtrError
from rbtr.oauth import oauth_is_set
from rbtr.providers import endpoint as endpoint_provider, model_context_window
from rbtr.styles import STYLE_DIM, STYLE_ERROR, STYLE_WARNING

from .compact import compact_history
from .connect import cmd_connect
from .draft_cmd import cmd_draft
from .index_cmd import cmd_index
from .llm import handle_llm
from .model import cmd_model, get_models
from .review import cmd_review
from .session import Session
from .shell import handle_shell
from .types import Command, TaskCancelled, TaskType


class Engine:
    """Executes commands and emits typed events. Knows nothing about the UI."""

    def __init__(self, session: Session, events: queue.Queue[Event]) -> None:
        self.session = session
        self._events = events
        self._last_shell_full_output: tuple[str, str, int, int] | None = (
            None  # (stdout, stderr, returncode, hidden_count)
        )
        self._cancel = threading.Event()
        self._shell_proc: subprocess.Popen[str] | None = None
        self._shell_pgid: int | None = None
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
        self._events.put(event)

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
        except Exception as e:
            self._error(f"Unexpected error: {e}")
            success = False
        self._emit(TaskFinished(success=success, cancelled=cancelled))

    # ── Setup ────────────────────────────────────────────────────────

    def _run_setup(self) -> None:
        # Deferred: open_repo calls pygit2.discover_repository which needs CWD set.
        from rbtr.git import open_repo, parse_github_remote

        try:
            repo = open_repo()
            owner, repo_name = parse_github_remote(repo)
        except RbtrError as e:
            self._error(str(e))
            return

        self.session.repo = repo
        self.session.owner = owner
        self.session.repo_name = repo_name
        self._out(f"Repository: {owner}/{repo_name}")

        if creds.github_token:
            gh = Github(auth=Auth.Token(creds.github_token), timeout=config.github.timeout)
            self.session.gh = gh
            self.session.gh_username = gh.get_user().login
            self._out("Authenticated with GitHub.")
        else:
            self._out("Not authenticated. Use /connect github to authenticate.")

        if oauth_is_set(creds.claude):
            self.session.claude_connected = True
            self._out("Connected to Anthropic.")

        if oauth_is_set(creds.chatgpt):
            self.session.chatgpt_connected = True
            self._out("Connected to ChatGPT.")

        if creds.openai_api_key:
            self.session.openai_connected = True
            self._out("Connected to OpenAI.")

        endpoints = endpoint_provider.list_endpoints()
        for ep in endpoints:
            self._out(f"Endpoint: {ep.name} ({ep.base_url})")

        # Pre-populate model cache so Tab completion is instant.
        get_models(self)

        if not (self.session.has_llm or endpoints):
            self._out("No LLM connected. Use /connect claude, chatgpt, or openai.")

        # Load saved model preference
        saved_model = config.model
        if saved_model:
            self.session.model_name = saved_model
            _init_context_window(self)

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
        self.session.message_history.clear()
        self.session.usage.reset()
        self._out("Conversation cleared.")

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
    ctx = model_context_window(engine.session.model_name)
    if ctx is not None:
        engine.session.usage.context_window = ctx
        engine.session.usage.context_window_known = True
