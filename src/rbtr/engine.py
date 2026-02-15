"""Engine — runs commands in daemon threads, emits Events.

Knows nothing about Rich, Live, or Panels. Never touches the display.
Communicates with the UI through ``queue.Queue[Event]`` using models
defined in ``rbtr.events``.
"""

import contextlib
import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import pygit2
from github import Auth, Github
from github.GithubException import GithubException

from rbtr import RbtrError
from rbtr.constants import (
    GITHUB_TIMEOUT,
    SHELL_MAX_LINES,
)
from rbtr.events import (
    ColumnDef,
    Event,
    FlushPanel,
    LinkOutput,
    MarkdownOutput,
    Output,
    TableOutput,
    TaskFinished,
    TaskStarted,
)
from rbtr.github import auth, client
from rbtr.models import PRSummary
from rbtr.repo import list_local_branches
from rbtr.styles import (
    CODE_HIGHLIGHT,
    COLUMN_BRANCH,
    LINK_STYLE,
    STYLE_DIM,
    STYLE_ERROR,
    STYLE_SHELL_STDERR,
    STYLE_WARNING,
)

_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/review": "List open PRs, or select a target: /review [pr_number|branch_name]",
    "/login": "Authenticate with GitHub",
    "/quit": "Exit rbtr (also /q)",
}


# ═══════════════════════════════════════════════════════════════════════
# Session state — shared between engine and UI (read by UI, written by engine)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Session:
    """Mutable state for the current rbtr session."""

    repo: pygit2.Repository | None = None
    owner: str = ""
    repo_name: str = ""
    gh: Github | None = None
    pr: PRSummary | None = None


# ═══════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════


class _TaskCancelled(Exception):
    """Raised inside a task thread when the user requests cancellation."""


class Engine:
    """Executes commands and emits typed events. Knows nothing about the UI."""

    def __init__(
        self, session: Session, events: queue.Queue[Event], pr_number: int | None = None
    ) -> None:
        self.session = session
        self._events = events
        self._pr_number = pr_number
        self._last_shell_full_output: tuple[str, str, int, int] | None = (
            None  # (stdout, stderr, returncode, hidden_count)
        )
        self._cancel = threading.Event()
        self._shell_proc: subprocess.Popen[str] | None = None
        self._shell_pgid: int | None = None

    # ── Cancellation ─────────────────────────────────────────────────

    def cancel(self) -> None:
        """Signal the running task to stop. Thread-safe."""
        self._cancel.set()
        pgid = self._shell_pgid
        if pgid is not None:
            with contextlib.suppress(OSError):
                os.killpg(pgid, signal.SIGTERM)

    def _check_cancel(self) -> None:
        """Raise _TaskCancelled if cancellation was requested."""
        if self._cancel.is_set():
            raise _TaskCancelled

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
        """Flush current output as a completed panel and start fresh.

        Only for slash-command handlers. Shell commands always produce
        a single panel — never call this from ``_handle_shell``.
        """
        self._emit(FlushPanel())

    def _clear(self) -> None:
        """Discard current transient output (e.g. "Fetching…" messages).

        Only for slash-command handlers. Shell commands always produce
        a single panel — never call this from ``_handle_shell``.
        """
        self._emit(FlushPanel(discard=True))

    # ── Task runner ──────────────────────────────────────────────────

    def run_task(self, task_type: str, arg: str) -> None:
        """Run a task synchronously (called from a daemon thread)."""
        self._emit(TaskStarted(task_id=f"{task_type}:{arg}"))
        success = True
        cancelled = False
        try:
            if task_type == "setup":
                self._run_setup()
            elif task_type == "command":
                self._handle_command(arg)
            elif task_type == "shell":
                self._handle_shell(arg)
            elif task_type == "llm":
                self._handle_llm(arg)
        except _TaskCancelled:
            success = False
            cancelled = True
        except Exception as e:
            self._error(f"Unexpected error: {e}")
            success = False
        self._emit(TaskFinished(success=success, cancelled=cancelled))

    # ── Setup ────────────────────────────────────────────────────────

    def _run_setup(self) -> None:
        from rbtr.repo import open_repo, parse_github_remote

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

        token = auth.load_token()
        if token:
            self.session.gh = Github(auth=Auth.Token(token), timeout=GITHUB_TIMEOUT)
            self._out("Authenticated with GitHub.")
        else:
            self._out("Not authenticated. Use /login to authenticate.")

        self._out("Type a message for the LLM, /help for commands, !cmd for shell")

    # ── Commands ─────────────────────────────────────────────────────

    def _handle_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            self._cmd_help()
        elif cmd == "/review":
            self._cmd_review(args)
        elif cmd == "/login":
            self._cmd_login()
        else:
            self._warn(f"Unknown command: {cmd}. Type /help for commands.")

    def _cmd_help(self) -> None:
        for name, desc in _COMMANDS.items():
            self._emit(Output(text=f"  {name:<12}{desc}", style=STYLE_DIM))

    def _cmd_review(self, identifier: str) -> None:
        if not identifier:
            self._list()
            return
        try:
            pr_number = int(identifier)
            self._review_pr(pr_number)
            return
        except ValueError:
            pass
        self._review_branch(identifier)

    def _cmd_login(self) -> None:
        auth.clear_token()
        try:
            device_response = auth.request_device_code()
            user_code = device_response["user_code"]
            verification_uri = device_response["verification_uri"]
            device_code = device_response["device_code"]
            interval = int(device_response.get("interval", "5"))

            self._emit(
                LinkOutput(
                    markup=(
                        f"Open [link={verification_uri}][{LINK_STYLE}]{verification_uri}"
                        f"[/{LINK_STYLE}][/link] and enter code: "
                        f"[{CODE_HIGHLIGHT}]{user_code}[/{CODE_HIGHLIGHT}]"
                    )
                )
            )
            self._copy_to_clipboard(user_code)
            self._out("Code copied to clipboard.")
            self._flush()

            self._out("Waiting for authorization…")
            token = auth.poll_for_token(device_code, interval)
            self._clear()

            auth.save_token(token)
            self.session.gh = Github(auth=Auth.Token(token), timeout=GITHUB_TIMEOUT)
            self._out("Authenticated with GitHub.")
        except Exception as e:
            self._warn(f"Authentication failed: {e}")

    # ── Listing ──────────────────────────────────────────────────────

    def _list(self) -> None:
        if self.session.gh is not None:
            try:
                self._out("Fetching from GitHub…")
                prs = client.list_open_prs(
                    self.session.gh, self.session.owner, self.session.repo_name
                )
                self._check_cancel()
                pr_branches = {pr.head_branch for pr in prs}
                branches = client.list_unmerged_branches(
                    self.session.gh,
                    self.session.owner,
                    self.session.repo_name,
                    pr_branches,
                )
                self._clear()

                if not prs and not branches:
                    self._out(
                        f"No open PRs or unmerged branches in "
                        f"{self.session.owner}/{self.session.repo_name}."
                    )
                    return

                if prs:
                    self._emit(
                        TableOutput(
                            title="Open Pull Requests",
                            columns=[
                                ColumnDef(header="PR", width=8),
                                ColumnDef(header="Title"),
                                ColumnDef(header="Author", width=16),
                                ColumnDef(header="Branch", style=COLUMN_BRANCH),
                            ],
                            rows=[
                                [f"#{pr.number}", pr.title, pr.author, pr.head_branch] for pr in prs
                            ],
                        )
                    )

                if prs and branches:
                    self._flush()

                if branches:
                    self._emit(
                        TableOutput(
                            title="Unmerged Branches",
                            columns=[
                                ColumnDef(header="Branch", style=COLUMN_BRANCH),
                                ColumnDef(header="Last Commit"),
                                ColumnDef(header="Updated"),
                            ],
                            rows=[
                                [
                                    b.name,
                                    b.last_commit_message[:60],
                                    b.updated_at.strftime("%Y-%m-%d"),
                                ]
                                for b in branches
                            ],
                        )
                    )

                self._out("Use /review <pr_number> or /review <branch_name> to select.")
                return
            except GithubException as e:
                self._clear()
                if e.status in (403, 404):
                    self._warn_access(e)
                else:
                    self._warn(f"GitHub error ({e.status}): {e.data}")
                    self._out("Falling back to local branches.")

        if self.session.repo is None:
            return
        branches_local = list_local_branches(self.session.repo)
        if not branches_local:
            self._out("No local branches found.")
            return

        self._emit(
            TableOutput(
                title="Local Branches",
                columns=[
                    ColumnDef(header="Branch", style=COLUMN_BRANCH),
                    ColumnDef(header="Last Commit"),
                    ColumnDef(header="Updated"),
                ],
                rows=[
                    [
                        b.name,
                        b.last_commit_message[:60],
                        b.updated_at.strftime("%Y-%m-%d"),
                    ]
                    for b in branches_local
                ],
            )
        )
        self._out("Use /review <branch_name> to select.")

    # ── Review targets ───────────────────────────────────────────────

    def _review_pr(self, pr_number: int) -> None:
        if self.session.gh is None:
            self._warn("Not authenticated. Run /login first.")
            return
        try:
            self._out(f"Fetching PR #{pr_number}…")
            pr = client.validate_pr_number(
                self.session.gh, self.session.owner, self.session.repo_name, pr_number
            )
            self._clear()
            self.session.pr = pr
            self._print_review_target()
        except GithubException as e:
            self._clear()
            self._warn(f"Could not fetch PR #{pr_number}: {e.data}")
        except RbtrError as e:
            self._clear()
            self._warn(str(e))

    def _review_branch(self, name: str) -> None:
        if self.session.repo is None:
            return
        for branch_name in self.session.repo.branches.local:
            if branch_name == name:
                branch = self.session.repo.branches.local[branch_name]
                commit = branch.peel(pygit2.Commit)
                self.session.pr = PRSummary(
                    title=name,
                    author="",
                    head_branch=name,
                    updated_at=datetime.fromtimestamp(commit.commit_time, tz=UTC),
                )
                self._print_review_target()
                return
        self._warn(f"'{name}' not found as a PR number or local branch.")

    def _print_review_target(self) -> None:
        if self.session.pr is None:
            self._out("No review target selected. Use /review to select one.")
            return
        pr = self.session.pr
        if pr.number is not None:
            self._out(f"Reviewing PR #{pr.number}: {pr.title} ({pr.head_branch})")
        else:
            self._out(f"Reviewing branch: {pr.head_branch}")

    def _warn_access(self, exc: GithubException) -> None:
        self._warn(
            f"Cannot access {self.session.owner}/{self.session.repo_name} "
            f"via GitHub API ({exc.status})."
        )
        message = exc.data.get("message", "") if isinstance(exc.data, dict) else ""
        if message:
            self._out(message)
        self._out("Falling back to local branches.")

    # ── Shell ────────────────────────────────────────────────────────

    @staticmethod
    def _truncate_output(text: str, max_lines: int) -> tuple[str, int]:
        """Truncate text to max_lines. Returns (truncated, hidden_count)."""
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text, 0
        truncated = "\n".join(lines[:max_lines])
        return truncated, len(lines) - max_lines

    def _handle_shell(self, cmd: str) -> None:
        if not cmd:
            self._out("Usage: !<command>")
            return
        self._out(f"$ {cmd}")
        # Give the child a pty as stdin so terminal ioctls work
        # (e.g. watch, less), but our reader thread keeps exclusive
        # access to the real stdin.
        master_fd, slave_fd = os.openpty()
        try:
            proc = subprocess.Popen(  # noqa: S602
                cmd,
                shell=True,
                stdin=slave_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)
        self._shell_proc = proc
        try:
            self._shell_pgid = os.getpgid(proc.pid)
        except OSError:
            self._shell_pgid = proc.pid
        try:
            # communicate(timeout) drains pipes properly (unlike wait).
            # First call starts internal reader threads; subsequent calls
            # just re-join them — safe to call in a loop.
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=0.02)
                    break
                except subprocess.TimeoutExpired:
                    self._check_cancel()
        except _TaskCancelled:
            # SIGKILL the entire process group — SIGTERM from cancel()
            # may have been trapped by the child (e.g. watch, less).
            with contextlib.suppress(OSError):
                os.killpg(self._shell_pgid, signal.SIGKILL)
            with contextlib.suppress(OSError):
                proc.wait(timeout=2)
            raise
        finally:
            self._shell_proc = None
            self._shell_pgid = None
            os.close(master_fd)
        self._check_cancel()
        stdout_full = stdout.rstrip() if stdout else ""
        stderr_full = stderr.rstrip() if stderr else ""
        total_hidden = 0
        if stdout_full:
            shown, hidden = self._truncate_output(stdout_full, SHELL_MAX_LINES)
            self._out(shown)
            total_hidden += hidden
        if stderr_full:
            shown, hidden = self._truncate_output(stderr_full, SHELL_MAX_LINES)
            self._out(shown, style=STYLE_SHELL_STDERR)
            total_hidden += hidden
        had_error = proc.returncode != 0
        if had_error and not self._cancel.is_set():
            self._error(f"(exit code {proc.returncode})")
        if total_hidden:
            self._last_shell_full_output = (stdout_full, stderr_full, proc.returncode, total_hidden)
        else:
            self._last_shell_full_output = None

    # ── LLM ──────────────────────────────────────────────────────────

    def _handle_llm(self, message: str) -> None:
        self._emit(MarkdownOutput(text="*LLM not connected yet. Agent integration coming soon.*"))

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
