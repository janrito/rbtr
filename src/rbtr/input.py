"""Input handling for rbtr, powered by prompt_toolkit's parser and buffer.

prompt_toolkit is used headlessly — it parses keystrokes and manages the
editing buffer.  All rendering goes through Rich in ``tui.py``.

Architecture
~~~~~~~~~~~~
::

    stdin ──▶ Vt100Parser ──▶ _on_key() ──▶ Buffer (editing state)
                                   │
                                   └──▶ InputState flags (submit, cancel, …)

    Main thread reads InputState + Buffer for Rich rendering.
"""

from __future__ import annotations

import contextlib
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import typing
from dataclasses import dataclass, field
from importlib import resources

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

from rbtr.config import config
from rbtr.constants import RBTR_DIR

HISTORY_PATH = RBTR_DIR / "history"

# (value, description) — a single completion candidate.
type Completions = list[tuple[str, str]]

# ── Pure helpers (no I/O, no state) ──────────────────────────────────


def replace_shell_word(text: str, replacement: str) -> str:
    """Replace the completable part of the last word in a shell command.

    If the last word is a path (contains ``/``) and the replacement
    looks like just a filename (no ``/`` prefix matching the original),
    the directory prefix from the original word is preserved::

        replace_shell_word("ls src/rb", "rbtr/")  → "ls src/rbtr/"
        replace_shell_word("git sta", "status")     → "git status"
    """
    token_start = len(text)
    while token_start > 0 and text[token_start - 1] != " ":
        token_start -= 1
    original_word = text[token_start:]

    if "/" in original_word and not replacement.startswith(original_word.rsplit("/", 1)[0] + "/"):
        dir_prefix = original_word.rsplit("/", 1)[0] + "/"
        return text[:token_start] + dir_prefix + replacement

    return text[:token_start] + replacement


def complete_path(word: str) -> Completions:
    """Complete a filesystem path.  Returns ``(full_path, "")`` pairs.

    Values include the full path so callers can use them as direct
    replacements for the word being completed.

    Handles ``~`` (home directory), absolute paths, and relative paths.
    Hidden entries (starting with ``.``) are only shown when *word*
    itself starts with a dot (or contains ``/.``).
    """
    # Expand ~ to the real home directory for filesystem operations,
    # but keep the original prefix for display values.
    expanded = os.path.expanduser(word) if word.startswith("~") else word

    if not expanded:
        dir_path, partial = ".", ""
        display_dir = ""
    elif expanded.endswith("/"):
        dir_path = expanded.rstrip("/") or "/"
        partial = ""
        display_dir = word.rstrip("/") + "/"  # preserve ~/... in display
    elif "/" in expanded:
        dir_path, partial = expanded.rsplit("/", 1)
        dir_path = dir_path or "/"
        display_prefix, _ = word.rsplit("/", 1)
        display_dir = display_prefix + "/"
    else:
        dir_path, partial = ".", expanded
        display_dir = ""

    try:
        entries = os.listdir(dir_path)
    except OSError:
        return []

    show_hidden = partial.startswith(".")

    matches: Completions = []
    for entry in sorted(entries):
        if not entry.startswith(partial):
            continue
        if entry.startswith(".") and not show_hidden:
            continue
        full = display_dir + entry
        if os.path.isdir(os.path.join(dir_path, entry)):
            full += "/"
        matches.append((full, ""))
    return matches


def _bash_complete_script() -> str:
    """Return the filesystem path to ``bash_complete.sh``."""
    return str(resources.files("rbtr.scripts").joinpath("bash_complete.sh"))


def complete_bash(cmd_line: str) -> Completions:
    """Query bash's programmable completion for *cmd_line*.

    Searches well-known directories for bash-completion scripts, sources
    the appropriate one, calls the registered completion function with
    the ``COMP_*`` variables set, and returns ``COMPREPLY`` entries as
    ``(value, "")`` pairs.

    Returns an empty list when no completion script is found, the command
    has no registered completion function, or bash is not available.
    Timeout prevents runaway completions from blocking the UI.
    """
    # Deferred: input.py is a pure utility module — keep top-level imports minimal.
    import subprocess

    parts = cmd_line.split()
    if not parts:
        return []
    cmd = parts[0]

    # Trailing space → completing a new (empty) word.
    if cmd_line.endswith(" "):
        parts.append("")
    cword = len(parts) - 1

    script = _bash_complete_script()
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "bash",
                script,
                cmd,
                str(cword),
                cmd_line,
                *parts,
            ],
            capture_output=True,
            text=True,
            timeout=config.tui.shell_completion_timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    return [(line.rstrip(), "") for line in result.stdout.splitlines() if line.strip()]


def complete_executables(prefix: str) -> Completions:
    """Complete executable names from ``PATH``.

    Returns ``(name, "")`` pairs.  Includes exact matches (so ``git``
    appears even when ``git-latexdiff`` also matches).  Deterministic
    and fast — no subprocess or pty involved.
    """
    seen: set[str] = set()
    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        try:
            entries = os.listdir(dir_path)
        except OSError:
            continue
        for entry in entries:
            if entry.startswith(prefix) and entry not in seen:
                full = os.path.join(dir_path, entry)
                if os.access(full, os.X_OK):
                    seen.add(entry)
    return [(name, "") for name in sorted(seen)]


def query_shell_completions(
    cmd_line: str,
    max_results: int = config.tui.max_completions,
) -> Completions:
    """Return completions for a shell command line.

    Orchestrates three tiers of completion, falling through on empty
    results:

    1. **Bash programmable completion** — subcommands, flags, branch
       names, etc. from bash-completion scripts.
    2. **Filesystem path** — for arguments (non-first-word).
    3. **PATH executable search** — for command names (first word).

    Pure function — no UI, no Rich, no display side-effects.
    Safe to call from any thread.
    """
    if not cmd_line:
        return []

    parts = cmd_line.split()
    if cmd_line.endswith(" "):
        word = ""
        is_first_word = False
    else:
        word = parts[-1] if parts else ""
        is_first_word = len(parts) <= 1

    # Tier 1: bash programmable completion.
    matches = complete_bash(cmd_line)

    # Tier 2: filesystem path (for arguments only).
    if not matches and not is_first_word:
        matches = complete_path(word)

    # Tier 3: PATH executable search (for command names only).
    if not matches and is_first_word:
        matches = complete_executables(word)

    return matches[:max_results]


def shell_context_word(cmd_line: str) -> str:
    """Return the last whitespace-delimited word from a shell command line."""
    parts = cmd_line.split()
    return parts[-1] if parts else ""


def completion_suffix(value: str, context_word: str = "") -> str:
    """Return the suffix to append after accepting a completion.

    Follows standard shell convention:
    - directories          → "/" (user continues the path)
    - existing files       → " " (word is complete)
    - path-like but absent → "" (ambiguous partial — don't commit)
    - non-path tokens      → " " (commands/options are always complete)

    *context_word* is the original word being completed (from the buffer).
    If either the value or context contains ``/``, path semantics apply.
    """
    if value.endswith("/"):
        return ""
    if os.path.isdir(value):
        return "/"
    is_path = "/" in value or "/" in context_word
    if is_path:
        return " " if os.path.exists(value) else ""
    return " "


# ── InputState — shared between reader thread and UI main loop ───────


@dataclass
class InputState:
    """Observable input state.  The UI reads fields for rendering;
    the reader thread writes them from ``_on_key``."""

    buffer: Buffer = field(default_factory=lambda: Buffer(multiline=True))
    submitted: queue.Queue[str | None] = field(default_factory=queue.Queue)
    quit: bool = False
    cancel_requested: bool = False
    expand_requested: bool = False
    tab_pressed: bool = False
    last_ctrl_c: float = 0.0
    # Completion state — written by UI, cycled by reader.
    completions: Completions = field(default_factory=list)
    completion_index: int = -1
    # Set by UI so the reader knows whether Ctrl-C should cancel.
    active_task: bool = False
    # Called directly from the reader thread on Ctrl-C — bypasses
    # the main loop polling cycle for immediate cancellation.
    on_cancel: typing.Callable[[], None] | None = None
    # Shared history — reader uses for Up/Down, UI appends for
    # auto-dispatched commands (e.g. startup /review).
    history: list[str] = field(default_factory=list)

    # ── Convenience accessors ────────────────────────────────────────

    @property
    def text(self) -> str:
        """Current buffer text (shorthand for ``buffer.document.text``)."""
        return self.buffer.text

    @property
    def cursor(self) -> int:
        """Current cursor offset (shorthand for ``buffer.cursor_position``)."""
        return self.buffer.cursor_position

    def reset(self) -> None:
        """Clear the buffer and move cursor to position 0."""
        self.buffer.set_document(Document(), bypass_readonly=True)

    def set_text(self, text: str, cursor: int | None = None) -> None:
        """Replace buffer contents and optionally set cursor position."""
        pos = len(text) if cursor is None else min(cursor, len(text))
        self.buffer.set_document(Document(text, pos), bypass_readonly=True)

    def append_history(self, entry: str) -> None:
        """Append to in-memory list and flush to disk immediately."""
        self.history.append(entry)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        with contextlib.suppress(OSError):
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_PATH.write_text("\n".join(self.history) + "\n")

    def clear_completions(self) -> None:
        self.completions = []
        self.completion_index = -1

    def accept_completion(self, value: str, *, with_suffix: bool = False) -> None:
        """Replace the appropriate part of the buffer with *value*."""
        text = self.buffer.text
        if text.startswith("!"):
            cmd = text[1:]
            ctx = shell_context_word(cmd)
            new_cmd = replace_shell_word(cmd, value)
            suffix = completion_suffix(value, ctx) if with_suffix else ""
            new = "!" + new_cmd + suffix
        elif text.startswith("/"):
            suffix = " " if with_suffix else ""
            new = value + suffix
        else:
            new = value
        self.set_text(new)

    def apply_completions(self, matches: Completions) -> None:
        """Apply a list of ``(value, description)`` matches.

        - Single match → auto-accept (replace word + append suffix).
        - Multiple matches → extend common prefix, show menu.
        - No matches → clear menu.

        Truncates to ``config.tui.max_completions`` before displaying.
        """
        if len(matches) == 1:
            self.accept_completion(matches[0][0], with_suffix=True)
            self.clear_completions()
        elif matches:
            # Compute common prefix from ALL matches before truncating
            # so we don't mislead (e.g. extending to "c" when only
            # claude/chatgpt are in the first page but deepinfra exists).
            values = [m[0] for m in matches]
            prefix = os.path.commonprefix(values)
            text = self.buffer.text
            if text.startswith("/"):
                if len(prefix) > len(text):
                    self.set_text(prefix)
            else:
                cmd_line = text[1:]
                parts = cmd_line.split()
                original_word = parts[-1] if parts else ""
                completing = (
                    original_word.rsplit("/", 1)[-1] if "/" in original_word else original_word
                )
                if len(prefix) > len(completing):
                    self.set_text("!" + replace_shell_word(cmd_line, prefix))
            self.completions = matches[: config.tui.max_completions]
            self.completion_index = -1
        else:
            self.clear_completions()


# ── InputReader — daemon thread, owns terminal mode ──────────────────


class InputReader:
    """Reads keystrokes via ``Vt100Parser`` and drives a ``Buffer``.

    Use as a context manager — it sets cbreak mode on entry and
    restores the terminal on exit.  History is written to disk on
    every submit so a crash never loses it.
    """

    def __init__(self, state: InputState) -> None:
        self._state = state
        self._parser = Vt100Parser(self._on_key)
        self._fd = sys.stdin.fileno()
        self._old_settings: list[int | list[bytes | int]] | None = None
        self._old_sigint: object = signal.SIG_DFL
        self._thread: threading.Thread | None = None
        self._history_index: int = -1
        self._saved_text: str = ""
        self._search_prefix: str = ""
        # Escape tracking for Alt+key combinations.
        self._escape_next: bool = False

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self) -> InputReader:
        self._old_settings = termios.tcgetattr(self._fd)
        # Single atomic termios call — cbreak mode with ISIG disabled.
        # tty.setcbreak() leaves ISIG enabled (two-step was racy).
        mode = termios.tcgetattr(self._fd)
        mode[3] = mode[3] & ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
        mode[6][termios.VMIN] = 1
        mode[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSANOW, mode)
        # Safety net: if SIGINT arrives despite ISIG being off, treat
        # it as a cancel request instead of crashing with KeyboardInterrupt.
        self._old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)
        self._load_history()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        signal.signal(signal.SIGINT, self._old_sigint)  # type: ignore[arg-type]  # restoring saved handler
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def _sigint_handler(self, _sig: int, _frame: object) -> None:
        """Handle stray SIGINT — treat it as Ctrl+C."""
        state = self._state
        if state.on_cancel is not None:
            state.on_cancel()
        state.cancel_requested = True

    # ── History persistence ──────────────────────────────────────────

    def _load_history(self) -> None:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError, OSError):
            lines = HISTORY_PATH.read_text().splitlines()
            self._state.history = [ln for ln in lines if ln.strip()]

    # ── Reader loop ──────────────────────────────────────────────────

    def _loop(self) -> None:
        """Blocking read loop — runs in a daemon thread.

        Uses ``os.read`` on the raw fd rather than ``sys.stdin.read``
        so that ``select`` and ``read`` operate on the same kernel
        buffer.  This ensures escape sequences (e.g. ``\\x1b[B`` for
        Down) arrive as a single chunk and are parsed correctly.
        """
        fd = self._fd
        while not self._state.quit:
            try:
                # select with a short timeout so we can flush the parser
                # when stdin goes quiet (standalone Escape detection).
                ready, _, _ = select.select([fd], [], [], 0.02)
                if ready:
                    data = os.read(fd, 1024)
                    if not data:
                        self._state.submitted.put(None)
                        break
                    self._parser.feed(data.decode("utf-8", errors="replace"))
                else:
                    self._parser.flush()
            except (OSError, ValueError):
                break

    # ── Key dispatch ─────────────────────────────────────────────────

    def _on_key(self, key_press: KeyPress) -> None:
        key = key_press.key
        data: str = key_press.data
        state = self._state
        buf = state.buffer

        # ── Alt+key combos (Escape prefix) ───────────────────────────
        if self._escape_next:
            self._escape_next = False
            if self._handle_alt(key, buf, state):
                return
            # Not a recognized Alt combo — fall through and handle
            # the key normally (e.g. Escape then Ctrl-C → Ctrl-C).

        if key == Keys.Escape:
            self._escape_next = True
            return

        # ── Ctrl keys / custom actions ───────────────────────────────

        if key == Keys.ControlC:
            now = time.monotonic()
            # Double Ctrl+C always exits, whether a task is active or not.
            if now - state.last_ctrl_c < config.tui.double_ctrl_c_window:
                state.submitted.put(None)
                state.quit = True
                return
            state.last_ctrl_c = now
            # Call cancel directly from the reader thread — no
            # round-trip through the main loop polling cycle.
            if state.on_cancel is not None:
                state.on_cancel()
            state.cancel_requested = True
            if not state.active_task:
                state.reset()
                state.clear_completions()
            return

        if key == Keys.ControlD:
            state.submitted.put(None)
            state.quit = True
            return

        if key == Keys.ControlO:
            state.expand_requested = True
            return

        # ── Completion cycling (Up/Down while menu is open) ──────────

        if state.completions:
            if key == Keys.Down:
                if state.completion_index < len(state.completions) - 1:
                    state.completion_index += 1
                return
            if key == Keys.Up:
                if state.completion_index > 0:
                    state.completion_index -= 1
                return

        # ── Enter → submit ───────────────────────────────────────────

        if key in (Keys.Enter, Keys.ControlM, Keys.ControlJ):
            if state.completions and state.completion_index >= 0:
                selected = state.completions[state.completion_index][0]
                state.accept_completion(selected)
                state.clear_completions()
            text = buf.text.strip()
            state.reset()
            state.clear_completions()
            self._history_index = -1
            if text:
                state.append_history(text)
                state.submitted.put(text)
            return

        # ── Tab → completion ─────────────────────────────────────────

        if key == Keys.Tab:
            if state.completions and state.completion_index >= 0:
                selected = state.completions[state.completion_index][0]
                state.accept_completion(selected, with_suffix=True)
                state.clear_completions()
            else:
                state.tab_pressed = True
            return

        # ── Bracketed paste ──────────────────────────────────────────

        if key == Keys.BracketedPaste:
            buf.insert_text(data)
            state.clear_completions()
            return

        # ── Navigation ───────────────────────────────────────────────

        if key == Keys.Left:
            buf.cursor_left()
            return

        if key == Keys.Right:
            buf.cursor_right()
            return

        if key == Keys.Up:
            if buf.document.cursor_position_row > 0:
                buf.cursor_up()
            else:
                self._history_up()
            return

        if key == Keys.Down:
            if buf.document.cursor_position_row < buf.document.line_count - 1:
                buf.cursor_down()
            else:
                self._history_down()
            return

        if key in (Keys.Home, Keys.ControlA):
            buf.cursor_position += buf.document.get_start_of_line_position()
            return

        if key in (Keys.End, Keys.ControlE):
            buf.cursor_position += buf.document.get_end_of_line_position()
            return

        if key == Keys.ControlLeft:
            self._word_left(buf)
            return

        if key == Keys.ControlRight:
            self._word_right(buf)
            return

        # ── Editing ──────────────────────────────────────────────────

        if key in (Keys.Backspace, Keys.ControlH):
            buf.delete_before_cursor()
            state.clear_completions()
            return

        if key == Keys.Delete:
            buf.delete()
            state.clear_completions()
            return

        if key == Keys.ControlW:
            pos = buf.document.find_previous_word_beginning()
            if pos:
                buf.delete_before_cursor(-pos)
            state.clear_completions()
            return

        if key == Keys.ControlU:
            col = buf.document.cursor_position_col
            if col > 0:
                buf.delete_before_cursor(col)
            state.clear_completions()
            return

        if key == Keys.ControlK:
            end_pos = buf.document.get_end_of_line_position()
            if end_pos > 0:
                buf.delete(end_pos)
            state.clear_completions()
            return

        if key == Keys.ControlUnderscore:
            buf.undo()
            return

        # ── Printable characters ─────────────────────────────────────

        if isinstance(key, str) and len(key) == 1 and key.isprintable():
            buf.insert_text(key)
            state.clear_completions()
            return

    # ── Word navigation helpers ──────────────────────────────────────

    @staticmethod
    def _word_left(buf: Buffer) -> None:
        pos = buf.document.find_previous_word_beginning()
        if pos:
            buf.cursor_position += pos

    @staticmethod
    def _word_right(buf: Buffer) -> None:
        pos = buf.document.find_next_word_ending()
        if pos:
            buf.cursor_position += pos

    # ── Alt+key handler ──────────────────────────────────────────────

    def _handle_alt(self, key: object, buf: Buffer, state: InputState) -> bool:
        """Handle the key following an Escape prefix (Alt+key).

        Returns True if the combo was recognized and handled.
        Returning False lets the caller fall through to regular
        key handling — so e.g. Escape then Ctrl-C still cancels.
        """
        # Alt+Enter → insert newline
        if key in (Keys.Enter, Keys.ControlM, Keys.ControlJ):
            buf.insert_text("\n")
            state.clear_completions()
            return True

        # Alt+B / Alt+Left → word left
        if key in ("b", Keys.Left):
            self._word_left(buf)
            return True

        # Alt+F / Alt+Right → word right
        if key in ("f", Keys.Right):
            self._word_right(buf)
            return True

        # Alt+D → delete word forward
        if key == "d":
            pos = buf.document.find_next_word_ending()
            if pos:
                buf.delete(pos)
            state.clear_completions()
            return True

        return False

    # ── History navigation ───────────────────────────────────────────

    def _history_up(self) -> None:
        hist = self._state.history
        if not hist:
            return
        if self._history_index == -1:
            self._saved_text = self._state.buffer.text
            self._search_prefix = self._saved_text
            start = len(hist) - 1
        else:
            start = self._history_index - 1
        # Search backwards for a matching prefix.
        for i in range(start, -1, -1):
            if hist[i].startswith(self._search_prefix):
                self._history_index = i
                self._state.set_text(hist[i])
                return

    def _history_down(self) -> None:
        hist = self._state.history
        if self._history_index == -1:
            return
        # Search forwards for a matching prefix.
        for i in range(self._history_index + 1, len(hist)):
            if hist[i].startswith(self._search_prefix):
                self._history_index = i
                self._state.set_text(hist[i])
                return
        # No more matches — restore saved text.
        self._history_index = -1
        self._state.set_text(self._saved_text)
