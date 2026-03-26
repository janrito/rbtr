"""Tests for the completion pipeline.

- Pure-function tests for `replace_shell_word`, `completion_suffix`,
  `shell_context_word`.
- Unit tests for each completion tier: `complete_bash`,
  `complete_path`, `complete_executables`.
- Orchestrator test for `query_shell_completions` (tier fallthrough).
- Integration tests for `_apply_completions` and `_complete_shell`
  in the UI layer.
"""

import subprocess
import time
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from rbtr.tui.input import (
    InputState,
    complete_bash,
    complete_executables,
    complete_path,
    completion_suffix,
    query_shell_completions,
    replace_shell_word,
    shell_context_word,
)
from rbtr.tui.ui import UI

# ── replace_shell_word ──────────────────────────────────────────────


def test_replace_single_word() -> None:
    assert replace_shell_word("git", "grep") == "grep"


def test_replace_multi_word() -> None:
    assert replace_shell_word("git sta", "status") == "git status"


def test_replace_empty_string() -> None:
    assert replace_shell_word("", "status") == "status"


def test_replace_trailing_spaces_starts_new_word() -> None:
    assert replace_shell_word("git ", "log") == "git log"


def test_replace_three_words() -> None:
    assert replace_shell_word("docker compose u", "up") == "docker compose up"


def test_replace_path_preserves_prefix() -> None:
    assert replace_shell_word("ls src/rb", "rbtr/") == "ls src/rbtr/"


def test_replace_path_full_replacement() -> None:
    assert replace_shell_word("ls src/rb", "src/rbtr/") == "ls src/rbtr/"


def test_replace_path_at_start(input_state: InputState) -> None:
    assert replace_shell_word("src/rb", "rbtr/") == "src/rbtr/"


def test_replace_nested_path(input_state: InputState) -> None:
    assert replace_shell_word("cat src/rbtr/tu", "tui.py") == "cat src/rbtr/tui.py"


def test_replace_no_path_no_prefix(input_state: InputState) -> None:
    assert replace_shell_word("ls REA", "README.md") == "ls README.md"


# ── _apply_completions — slash commands ──────────────────────────────


def test_slash_single_match_auto_accepts(input_state: InputState) -> None:
    input_state.set_text("/hel")
    input_state.apply_completions([("/help", "Show help")])
    assert input_state.text == "/help "
    assert input_state.completions == []


def test_slash_multiple_matches_shows_menu(input_state: InputState) -> None:
    input_state.set_text("/re")
    matches = [("/review", "Review a PR"), ("/refresh", "Refresh")]
    input_state.apply_completions(matches)
    assert input_state.text == "/re"
    assert input_state.completions == matches
    assert input_state.completion_index == -1


def test_slash_multiple_matches_extends_prefix(input_state: InputState) -> None:
    input_state.set_text("/r")
    matches = [("/review", "Review a PR"), ("/refresh", "Refresh")]
    input_state.apply_completions(matches)
    assert input_state.text == "/re"


def test_slash_no_matches_clears(input_state: InputState) -> None:
    input_state.set_text("/xyz")
    input_state.apply_completions([])
    assert input_state.completions == []


# ── completion_suffix ────────────────────────────────────────────────


def test_suffix_regular_word_gets_space() -> None:
    assert completion_suffix("status") == " "


def test_suffix_option_gets_space() -> None:
    assert completion_suffix("--verbose") == " "


def test_suffix_trailing_slash_gets_nothing() -> None:
    assert completion_suffix("src/") == ""


def test_suffix_real_directory_gets_slash(tmp_path: Path) -> None:
    d = tmp_path / "mydir"
    d.mkdir()
    assert completion_suffix(str(d)) == "/"


def test_suffix_real_file_gets_space(tmp_path: Path) -> None:
    f = tmp_path / "myfile.txt"
    f.write_text("hello")
    assert completion_suffix(str(f)) == " "


def test_suffix_nonexistent_path_gets_empty() -> None:
    assert completion_suffix("/no/such/path/xyzzy") == ""


def test_suffix_nonexistent_non_path_gets_space(input_state: InputState) -> None:
    assert completion_suffix("xyzzy_nonexistent") == " "


def test_suffix_path_context_triggers_path_mode(input_state: InputState, tmp_path: Path) -> None:
    assert completion_suffix("partial_name", context_word="src/partial") == ""


def test_suffix_path_context_existing_file(input_state: InputState, tmp_path: Path) -> None:
    f = tmp_path / "real.txt"
    f.write_text("hello")
    assert completion_suffix(str(f), context_word=str(tmp_path) + "/re") == " "


# ── _apply_completions — shell commands (! prefix) ───────────────────


def test_shell_single_match_replaces_last_word(input_state: InputState) -> None:
    input_state.set_text("!git sta")
    input_state.apply_completions([("status", "")])
    assert input_state.text == "!git status "


def test_shell_single_match_first_word(input_state: InputState) -> None:
    input_state.set_text("!gi")
    input_state.apply_completions([("git", "")])
    assert input_state.text == "!git "


def test_shell_multiple_matches_extends_prefix(input_state: InputState) -> None:
    input_state.set_text("!git st")
    matches = [("status", ""), ("stash", "")]
    input_state.apply_completions(matches)
    assert input_state.text == "!git sta"
    assert input_state.completions == matches


def test_shell_multiple_no_prefix_extension(input_state: InputState) -> None:
    input_state.set_text("!git s")
    matches = [("status", ""), ("stash", ""), ("show", "")]
    input_state.apply_completions(matches)
    assert input_state.text == "!git s"
    assert input_state.completions == matches


def test_shell_single_match_directory_gets_slash(input_state: InputState, tmp_path: Path) -> None:
    d = tmp_path / "mydir"
    d.mkdir()
    input_state.set_text("!ls myd")
    input_state.apply_completions([(str(d), "")])
    assert input_state.text == f"!ls {d}/"
    assert input_state.completions == []


def test_shell_single_match_trailing_slash_no_double(input_state: InputState) -> None:
    input_state.set_text("!cd sr")
    input_state.apply_completions([("src/", "")])
    assert input_state.text == "!cd src/"
    assert input_state.completions == []


# ── query_shell_completions — orchestrator ───────────────────────────


def test_query_bash_is_preferred(mocker: MockerFixture) -> None:
    mocker.patch("rbtr.tui.input.complete_bash", return_value=[("status", ""), ("stash", "")])
    results = query_shell_completions("git sta")
    assert results == [("status", ""), ("stash", "")]


def test_query_path_fallback_for_later_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "api").mkdir()
    mocker.patch("rbtr.tui.input.complete_bash", return_value=[])
    results = query_shell_completions("ls ap")
    assert len(results) == 1
    assert results[0][0] == "api/"


def test_query_executable_fallback_for_first_word(mocker: MockerFixture) -> None:
    mocker.patch("rbtr.tui.input.complete_bash", return_value=[])
    mocker.patch("rbtr.tui.input.complete_executables", return_value=[("git", ""), ("gist", "")])
    results = query_shell_completions("gi")
    assert results == [("git", ""), ("gist", "")]


def test_query_empty_returns_empty() -> None:
    assert query_shell_completions("") == []


def test_query_max_results_caps_output(mocker: MockerFixture) -> None:
    many = [(f"cmd{i}", "") for i in range(50)]
    mocker.patch("rbtr.tui.input.complete_bash", return_value=many)
    results = query_shell_completions("x ", max_results=10)
    assert len(results) == 10


def test_query_no_path_fallback_for_first_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """First word uses executables, not filesystem paths."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    mocker.patch("rbtr.tui.input.complete_bash", return_value=[])
    mocker.patch("rbtr.tui.input.complete_executables", return_value=[])
    results = query_shell_completions("sr")
    assert results == []


# ── _complete_shell — UI threading / staleness ───────────────────────


def _wait_for_completion(ui: UI, timeout: float = 2.0) -> None:
    snapshot = ui.inp.text
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ui.inp.completions or ui.inp.text != snapshot:
            time.sleep(0.05)
            return
        time.sleep(0.02)


def test_complete_shell_applies_results(
    input_state: InputState, mocker: MockerFixture, headless_ui: UI
) -> None:
    input_state.set_text("!git sta")
    mocker.patch(
        "rbtr.tui.ui.query_shell_completions", return_value=[("status", ""), ("stash", "")]
    )
    headless_ui._complete_shell()
    _wait_for_completion(headless_ui)
    assert headless_ui.inp.completions == [("status", ""), ("stash", "")]


def test_complete_shell_does_not_block(
    input_state: InputState, mocker: MockerFixture, headless_ui: UI
) -> None:
    input_state.set_text("!git sta")

    def _slow_query(cmd_line: str, max_results: int = 20) -> list[tuple[str, str]]:
        time.sleep(5)
        return []

    mocker.patch(
        "rbtr.tui.ui.query_shell_completions",
        side_effect=_slow_query,
    )
    start = time.monotonic()
    headless_ui._complete_shell()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


def test_complete_shell_discards_stale_results(
    input_state: InputState, mocker: MockerFixture, headless_ui: UI
) -> None:
    input_state.set_text("!git sta")

    def slow_query(cmd_line: str, max_results: int = 20) -> list[tuple[str, str]]:
        time.sleep(0.1)
        return [("status", "")]

    mocker.patch("rbtr.tui.ui.query_shell_completions", side_effect=slow_query)
    headless_ui._complete_shell()
    headless_ui.inp.set_text("!git status --short")
    time.sleep(0.3)

    assert headless_ui.inp.text == "!git status --short"
    assert headless_ui.inp.completions == []


def test_complete_shell_empty_cmd_is_noop(input_state: InputState, headless_ui: UI) -> None:
    input_state.set_text("!")
    headless_ui._complete_shell()
    time.sleep(0.1)
    assert headless_ui.inp.completions == []


# ── shell_context_word ───────────────────────────────────────────────


def test_context_word_returns_last_word() -> None:
    assert shell_context_word("git status --short") == "--short"


def test_context_word_single_word() -> None:
    assert shell_context_word("git") == "git"


def test_context_word_empty_string() -> None:
    assert shell_context_word("") == ""


def test_context_word_trailing_space() -> None:
    assert shell_context_word("git status ") == "status"


# ── complete_path — filesystem path completion ───────────────────────


def test_path_lists_directory_contents(tmp_path: Path) -> None:
    (tmp_path / "foo.txt").write_text("x")
    (tmp_path / "bar").mkdir()
    results = complete_path(str(tmp_path) + "/")
    names = [r[0] for r in results]
    assert str(tmp_path / "bar") + "/" in names
    assert str(tmp_path / "foo.txt") in names


def test_path_partial_match(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("x")
    (tmp_path / "beta.py").write_text("x")
    results = complete_path(str(tmp_path) + "/al")
    assert len(results) == 1
    assert results[0][0] == str(tmp_path / "alpha.py")


def test_path_nested_directory(tmp_path: Path) -> None:
    nested = tmp_path / "api" / "start"
    nested.mkdir(parents=True)
    (nested / "start.sh").write_text("x")
    results = complete_path(str(tmp_path / "api" / "start") + "/")
    assert len(results) == 1
    assert results[0][0].endswith("start.sh")


def test_path_directory_gets_trailing_slash(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    results = complete_path(str(tmp_path) + "/")
    dirs = [r[0] for r in results if r[0].endswith("/")]
    assert any("subdir/" in d for d in dirs)


def test_path_hidden_files_skipped_by_default(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "visible").write_text("x")
    results = complete_path(str(tmp_path) + "/")
    names = [r[0] for r in results]
    assert not any(".hidden" in n for n in names)


def test_path_hidden_files_shown_when_dot_prefix(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_text("x")
    results = complete_path(str(tmp_path) + "/.")
    names = [r[0] for r in results]
    assert any(".hidden" in n for n in names)


def test_path_nonexistent_directory_returns_empty() -> None:
    results = complete_path("/no/such/path/xyzzy/")
    assert results == []


def test_path_empty_word_lists_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "afile.txt").write_text("x")
    monkeypatch.chdir(tmp_path)
    results = complete_path("")
    names = [r[0] for r in results]
    assert "afile.txt" in names


def test_path_values_are_full_paths(tmp_path: Path) -> None:
    """Completion values include the directory prefix."""
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "main.py").write_text("x")
    results = complete_path(str(sub) + "/")
    assert results[0][0] == str(sub / "main.py")


def test_path_tilde_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """~ expands to the home directory but display values keep ~."""
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "file.txt").write_text("x")
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path), 1))
    results = complete_path("~/")
    assert len(results) > 0
    assert all(r[0].startswith("~/") for r in results)
    names = [r[0] for r in results]
    assert any(n.startswith("~/D") for n in names)


def test_path_tilde_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """~/D completes to home directory entries starting with D."""
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "other").mkdir()
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", str(tmp_path), 1))
    results = complete_path("~/D")
    assert len(results) > 0
    assert all(r[0].startswith("~/D") for r in results)


def test_path_absolute_root(tmp_path: Path) -> None:
    """Absolute paths complete against the filesystem."""
    (tmp_path / "Users").mkdir()
    prefix = str(tmp_path / "Use")
    results = complete_path(prefix)
    assert len(results) == 1
    assert results[0][0] == str(tmp_path / "Users") + "/"


# ── complete_executables — PATH search ───────────────────────────────


def test_executables_finds_on_path(tmp_path: Path, mocker: MockerFixture) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "mycommand"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    mocker.patch.dict("os.environ", {"PATH": str(bin_dir)})
    results = complete_executables("myc")
    names = [r[0] for r in results]
    assert "mycommand" in names


def test_executables_includes_exact_match(tmp_path: Path, mocker: MockerFixture) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("git", "git-lfs", "git-latexdiff"):
        exe = bin_dir / name
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    mocker.patch.dict("os.environ", {"PATH": str(bin_dir)})
    results = complete_executables("git")
    names = [r[0] for r in results]
    assert "git" in names
    assert "git-lfs" in names


def test_executables_skips_non_executable(tmp_path: Path, mocker: MockerFixture) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "notexe").write_text("data")
    mocker.patch.dict("os.environ", {"PATH": str(bin_dir)})
    results = complete_executables("not")
    assert results == []


def test_executables_no_matches_returns_empty(tmp_path: Path, mocker: MockerFixture) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mocker.patch.dict("os.environ", {"PATH": str(bin_dir)})
    results = complete_executables("zzz_nonexistent")
    assert results == []


# ── complete_bash — bash programmable completion ─────────────────────

# Detect once whether bash-completion for git is available on this machine.
# We probe with a known-good query; if the framework isn't installed we
# skip the tests that depend on it — but if it IS installed and our code
# returns nothing, that's a real failure.
_GIT_BASH_COMPLETION_AVAILABLE = bool(complete_bash("git status"))

_needs_bash_completion = pytest.mark.skipif(
    not _GIT_BASH_COMPLETION_AVAILABLE,
    reason="bash-completion for git not installed",
)


@_needs_bash_completion
def test_bash_git_subcommands() -> None:
    results = complete_bash("git sta")
    names = [r[0].strip() for r in results]
    assert "status" in names


@_needs_bash_completion
def test_bash_git_flags() -> None:
    results = complete_bash("git status --")
    names = [r[0].strip() for r in results]
    assert any(n.startswith("--") for n in names)


def test_bash_no_completion_returns_empty() -> None:
    results = complete_bash("zzz_no_such_command_12345 ")
    assert results == []


def test_bash_empty_input_returns_empty() -> None:
    results = complete_bash("")
    assert results == []


def test_bash_timeout_does_not_raise(mocker: MockerFixture) -> None:
    """Even if bash hangs, we get an empty list (no exception)."""
    mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("bash", 2))
    results = complete_bash("git sta")
    assert results == []
