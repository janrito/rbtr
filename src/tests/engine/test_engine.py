"""End-to-end tests for the Engine via its event contract.

These tests create a real Engine with a mock EngineState, run tasks
synchronously (no daemon threads needed — run_task is a plain method),
and assert on the sequence of events emitted to the queue.
"""

import queue
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import pygit2
import pytest
from github.GithubException import GithubException
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.creds import OAuthCreds, creds
from rbtr.engine import Engine, TaskType
from rbtr.engine.shell import _truncate_output
from rbtr.events import (
    Event,
    FlushPanel,
    IndexProgress,
    IndexReady,
    IndexStarted,
    LinkOutput,
    Output,
    TableOutput,
    TaskFinished,
    TaskStarted,
    TextDelta,
)
from rbtr.exceptions import RbtrError, TaskCancelled
from rbtr.models import BranchTarget, PRSummary, PRTarget, SnapshotTarget
from rbtr.providers import BuiltinProvider
from rbtr.state import EngineState

from .conftest import (
    BRANCH_FEATURE_X,
    PR_ADD_FEATURE,
    PR_FIX_BUG,
    PR_REFACTOR,
    drain,
    has_event_type,
    output_texts,
)

# ── /help ────────────────────────────────────────────────────────────


def test_help_lists_all_commands(engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/help")
    drained_events = drain(engine.events)

    assert isinstance(drained_events[0], TaskStarted)
    assert isinstance(drained_events[-1], TaskFinished)
    assert drained_events[-1].success is True

    texts = output_texts(drained_events)
    commands_mentioned = [t for t in texts if "/help" in t or "/review" in t or "/quit" in t]
    assert len(commands_mentioned) >= 3


def test_unknown_command_warns(engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/nonexistent")
    drained_events = drain(engine.events)

    texts = output_texts(drained_events)
    assert any("Unknown command" in t for t in texts)
    assert drained_events[-1].success is True


# ── /review (list mode) ─────────────────────────────────────────────


def test_list_with_github_prs(mocker: MockerFixture, engine: Engine) -> None:
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", return_value=[PR_FIX_BUG])
    mocker.patch(
        "rbtr.engine.review_cmd.client.list_unmerged_branches", return_value=[BRANCH_FEATURE_X]
    )
    engine.run_task(TaskType.COMMAND, "/review")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert has_event_type(drained_events, TableOutput)

    tables = [e for e in drained_events if isinstance(e, TableOutput)]
    assert len(tables) == 2
    assert tables[0].title == "Open Pull Requests"
    assert tables[1].title == "Unmerged Branches"


def test_list_without_auth_falls_back_to_local(repo_engine: Engine) -> None:
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature-1", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    tables = [e for e in drained_events if isinstance(e, TableOutput)]
    assert len(tables) == 1
    assert tables[0].title == "Local Branches"


def test_list_no_prs_no_branches(mocker: MockerFixture, engine: Engine) -> None:
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", return_value=[])
    mocker.patch("rbtr.engine.review_cmd.client.list_unmerged_branches", return_value=[])
    engine.run_task(TaskType.COMMAND, "/review")

    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("No open PRs" in t for t in texts)


# ── /review completion cache ──────────────────────────────────────────


def test_list_caches_review_targets_github(mocker: MockerFixture, engine: Engine) -> None:
    """After listing, engine.state.cached_review_targets has PRs and branches."""
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", return_value=[PR_FIX_BUG])
    mocker.patch(
        "rbtr.engine.review_cmd.client.list_unmerged_branches", return_value=[BRANCH_FEATURE_X]
    )
    engine.run_task(TaskType.COMMAND, "/review")
    drain(engine.events)

    cached = engine.state.cached_review_targets
    assert len(cached) == 2
    labels = [label for label, _ in cached]
    texts = [text for _, text in cached]
    assert any("#1" in lbl for lbl in labels)
    assert "1" in texts
    assert "feature-x" in texts


def test_list_always_refetches(mocker: MockerFixture, engine: Engine) -> None:
    """Calling /review a second time refetches — never serves stale cache."""
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    prs_v1 = [
        PRSummary(
            number=1,
            title="Old PR",
            author="alice",
            base_branch="main",
            head_branch="old",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    ]
    prs_v2 = [
        PRSummary(
            number=2,
            title="New PR",
            author="bob",
            base_branch="main",
            head_branch="new",
            updated_at=datetime(2025, 2, 1, tzinfo=UTC),
        ),
    ]

    mock_list = mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", return_value=prs_v1)
    mocker.patch("rbtr.engine.review_cmd.client.list_unmerged_branches", return_value=[])
    engine.run_task(TaskType.COMMAND, "/review")
    drain(engine.events)
    assert engine.state.cached_review_targets[0][1] == "1"

    # Second call with updated data — must refetch.
    mock_list.return_value = prs_v2
    engine.run_task(TaskType.COMMAND, "/review")
    drain(engine.events)
    assert engine.state.cached_review_targets[0][1] == "2"
    assert mock_list.call_count == 2


def test_list_caches_review_targets_local(repo_engine: Engine) -> None:
    """After listing local branches, cached_review_targets is populated."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature-1", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review")
    drain(engine.events)

    cached = engine.state.cached_review_targets
    assert len(cached) == 1
    assert cached[0] == ("feature-1", "feature-1")


# ── /review <target> ────────────────────────────────────────────────


def test_review_pr_by_number(mocker: MockerFixture, engine: Engine) -> None:
    mocker.patch("rbtr.engine.review_cmd.run_index")
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.validate_pr_number", return_value=PR_ADD_FEATURE)
    engine.run_task(TaskType.COMMAND, "/review 42")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert isinstance(engine.state.review_target, PRTarget)
    assert engine.state.review_target.number == 42


def test_review_without_arg_lists(repo_engine: Engine) -> None:
    """/review with no argument behaves like the old /list."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature-x", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert has_event_type(drained_events, TableOutput)


def test_review_pr_without_auth_warns(engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/review 42")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Not authenticated" in t for t in texts)


def test_review_snapshot_by_branch(mocker: MockerFixture, repo_engine: Engine) -> None:
    """Single-arg /review creates a SnapshotTarget."""
    mocker.patch("rbtr.engine.review_cmd.run_index")
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("my-branch", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review my-branch")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert isinstance(engine.state.review_target, SnapshotTarget)
    assert engine.state.review_target.ref_label == "my-branch"


def test_review_snapshot_head(mocker: MockerFixture, repo_engine: Engine) -> None:
    """/review HEAD creates a SnapshotTarget at the current HEAD."""
    mocker.patch("rbtr.engine.review_cmd.run_index")
    engine = repo_engine

    engine.run_task(TaskType.COMMAND, "/review HEAD")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert isinstance(engine.state.review_target, SnapshotTarget)
    assert engine.state.review_target.ref_label == "HEAD"


def test_review_nonexistent_ref_warns(repo_engine: Engine) -> None:
    engine = repo_engine

    engine.run_task(TaskType.COMMAND, "/review ghost-branch")

    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("not found" in t for t in texts)


def test_review_branch_two_args(mocker: MockerFixture, repo_engine: Engine) -> None:
    """/review base target sets both branches explicitly."""
    mocker.patch("rbtr.engine.review_cmd.run_index")
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("develop", repo.head.peel(pygit2.Commit))
    repo.branches.local.create("feature-x", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review develop feature-x")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert isinstance(engine.state.review_target, BranchTarget)
    assert engine.state.review_target.base_branch == "develop"
    assert engine.state.review_target.head_branch == "feature-x"
    texts = output_texts(drained_events)
    assert any("develop" in t and "feature-x" in t for t in texts)


def test_review_branch_bad_base_warns(repo_engine: Engine) -> None:
    """/review nonexistent target warns about missing base."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature-x", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review nonexistent feature-x")

    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("nonexistent" in t and "not found" in t for t in texts)


def test_review_branch_too_many_args_warns(engine: Engine) -> None:
    """/review a b c shows usage."""
    engine.run_task(TaskType.COMMAND, "/review a b c")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Usage" in t for t in texts)


def test_review_pr_has_base_branch(mocker: MockerFixture, engine: Engine) -> None:
    """PR review populates base_branch from the PR data."""
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.validate_pr_number", return_value=PR_REFACTOR)
    engine.run_task(TaskType.COMMAND, "/review 10")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert isinstance(engine.state.review_target, PRTarget)
    assert engine.state.review_target.base_branch == "develop"
    assert engine.state.review_target.head_branch == "refactor-x"


# ── !shell ───────────────────────────────────────────────────────────


def test_shell_captures_stdout(engine: Engine) -> None:
    engine.run_task(TaskType.SHELL, "echo hello")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    texts = output_texts(drained_events)
    assert any("hello" in t for t in texts)


def test_shell_captures_nonzero_exit(engine: Engine) -> None:
    engine.run_task(TaskType.SHELL, "false")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True  # task itself succeeds; error is in output
    texts = output_texts(drained_events)
    assert any("exit code" in t for t in texts)


def test_shell_empty_shows_usage(engine: Engine) -> None:
    engine.run_task(TaskType.SHELL, "")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("Usage" in t for t in texts)


def test_shell_truncates_long_output(engine: Engine) -> None:
    engine.run_task(TaskType.SHELL, "seq 1 100")
    drained_events = drain(engine.events)
    output_texts(drained_events)
    assert engine._last_shell_full_output is not None
    stdout_full, _, _, hidden_count = engine._last_shell_full_output
    assert len(stdout_full.split("\n")) > config.tui.shell_max_lines
    assert hidden_count > 0


def test_shell_never_emits_flush_panel(engine: Engine) -> None:
    """Shell commands always produce a single panel — no FlushPanel."""
    engine.run_task(TaskType.SHELL, "echo hello && echo world")
    drained_events = drain(engine.events)
    assert not any(isinstance(e, FlushPanel) for e in drained_events)


def test_shell_with_stderr_never_emits_flush_panel(engine: Engine) -> None:
    """Even with mixed stdout/stderr, shell stays in one panel."""
    engine.run_task(TaskType.SHELL, "echo out; echo err >&2")
    drained_events = drain(engine.events)
    assert not any(isinstance(e, FlushPanel) for e in drained_events)


# ── LLM placeholder ─────────────────────────────────────────────────


def test_llm_streams_response(creds_path: Path, mocker: MockerFixture, engine: Engine) -> None:
    """LLM messages build a model and stream via the agent."""

    creds.update(claude=OAuthCreds(access_token="t", refresh_token="r", expires_at=9e9))
    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"

    async def fake_stream(ctx, model, message, **kwargs):
        from rbtr.events import TextDelta

        ctx.emit(TextDelta(delta="Hello "))
        ctx.emit(TextDelta(delta="world"))

    mocker.patch("rbtr.llm.stream._stream_agent", fake_stream)
    engine.run_task(TaskType.LLM, "explain this code")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    deltas = [e for e in drained_events if isinstance(e, TextDelta)]
    assert len(deltas) == 2
    assert deltas[0].delta == "Hello "
    assert deltas[1].delta == "world"


def test_llm_persists_to_store(creds_path: Path, mocker: MockerFixture, engine: Engine) -> None:
    """After an LLM call, messages are persisted to the store."""

    creds.update(openai_api_key="sk-test")
    engine.state.connected_providers.add(BuiltinProvider.OPENAI)
    engine.state.model_name = "openai/gpt-4o"

    async def fake_stream(ctx, model, message, **kwargs):
        ctx.store.save_messages(
            ctx.state.session_id,
            [ModelRequest(parts=[UserPromptPart(content=message)])],
        )

    mocker.patch("rbtr.llm.stream._stream_agent", fake_stream)
    engine.run_task(TaskType.LLM, "hello")
    drain(engine.events)
    assert len(engine.store.load_messages(engine.state.session_id)) == 1


def test_new_starts_fresh_session(engine: Engine) -> None:
    engine._sync_store_context()
    engine.store.save_messages(
        engine.state.session_id,
        [ModelRequest(parts=[UserPromptPart(content="old")])],
    )
    old_session_id = engine.state.session_id
    engine.run_task(TaskType.COMMAND, "/new")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    # New session ID — old messages untouched, new session is empty.
    assert engine.state.session_id != old_session_id
    assert engine.store.load_messages(engine.state.session_id) == []
    texts = output_texts(drained_events)
    assert any("cleared" in t.lower() for t in texts)


def test_llm_warns_when_not_connected(engine: Engine) -> None:
    engine.run_task(TaskType.LLM, "explain this code")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    warnings = [e for e in drained_events if isinstance(e, Output) and "No LLM connected" in e.text]
    assert len(warnings) == 1


# ── Setup ────────────────────────────────────────────────────────────


def test_setup_in_valid_repo(
    monkeypatch: pytest.MonkeyPatch, creds_path: Path, engine: Engine
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("Test", "test@test.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.remotes.create("origin", "git@github.com:testowner/testrepo.git")

        monkeypatch.chdir(tmp)
        engine.state.owner = ""
        engine.state.repo_name = ""

        engine.run_task(TaskType.SETUP, "")

        drained_events = drain(engine.events)
        assert drained_events[-1].success is True
        assert engine.state.owner == "testowner"
        assert engine.state.repo_name == "testrepo"
        texts = output_texts(drained_events)
        assert any("testowner/testrepo" in t for t in texts)
        assert any("Not authenticated" in t for t in texts)


def test_setup_outside_repo_errors(monkeypatch: pytest.MonkeyPatch, engine: Engine) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        engine.state.owner = ""
        engine.state.repo_name = ""

        engine.run_task(TaskType.SETUP, "")

        drained_events = drain(engine.events)
        texts = output_texts(drained_events)
        assert any("git repository" in t for t in texts)


# ── Event contract ───────────────────────────────────────────────────


def test_every_task_starts_and_finishes(engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/help")
    drained_events = drain(engine.events)

    assert isinstance(drained_events[0], TaskStarted)
    assert isinstance(drained_events[-1], TaskFinished)


def test_task_finished_reports_success(engine: Engine) -> None:
    engine.run_task(TaskType.COMMAND, "/help")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True


def test_no_events_outside_task_boundary(engine: Engine) -> None:
    """All Output events must be between TaskStarted and TaskFinished."""
    engine.run_task(TaskType.COMMAND, "/review")
    drained_events = drain(engine.events)

    started = False
    for e in drained_events:
        if isinstance(e, TaskStarted):
            started = True
        elif isinstance(e, TaskFinished):
            started = False
        else:
            assert started, f"Event {type(e).__name__} emitted outside task boundary"


# ── /connect github ───────────────────────────────────────────────────


def test_connect_github_success(creds_path: Path, mocker: MockerFixture, engine: Engine) -> None:

    device_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "device_code": "dev123",
        "interval": "0",
    }

    mocker.patch(
        "rbtr.engine.connect_cmd.github_auth.request_device_code", return_value=device_resp
    )
    mocker.patch("rbtr.engine.connect_cmd.github_auth.poll_for_token", return_value="ghp_newtoken")
    mocker.patch.object(Engine, "_copy_to_clipboard")

    # Mock Github so get_user().login succeeds without a real API call.
    fake_gh = mocker.MagicMock()
    fake_gh.get_user.return_value.login = "testuser"
    mocker.patch("rbtr.engine.connect_cmd.Github", return_value=fake_gh)

    engine.run_task(TaskType.COMMAND, "/connect github")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert engine.state.gh is not None
    assert engine.state.gh_username == "testuser"
    assert creds.github_token == "ghp_newtoken"

    links = [e for e in drained_events if isinstance(e, LinkOutput)]
    assert len(links) == 1
    assert "github.com/login/device" in links[0].url

    texts = output_texts(drained_events)
    assert any("ABCD-1234" in t for t in texts)
    assert any("Authenticated" in t for t in texts)


def test_connect_github_failure(creds_path: Path, mocker: MockerFixture, engine: Engine) -> None:

    mocker.patch(
        "rbtr.engine.connect_cmd.github_auth.request_device_code",
        side_effect=RbtrError("network error"),
    )
    engine.run_task(TaskType.COMMAND, "/connect github")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert engine.state.gh is None
    texts = output_texts(drained_events)
    assert any("failed" in t.lower() for t in texts)


# ── GitHub error fallbacks ───────────────────────────────────────────


def test_403_falls_back_to_local_branches(mocker: MockerFixture, repo_engine: Engine) -> None:
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature-1", repo.head.peel(pygit2.Commit))
    engine.state.gh = mocker.MagicMock()

    exc = GithubException(403, {"message": "Resource not accessible"}, None)
    mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", side_effect=exc)
    engine.run_task(TaskType.COMMAND, "/review")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    texts = output_texts(drained_events)
    assert any("Cannot access" in t for t in texts)
    tables = [e for e in drained_events if isinstance(e, TableOutput)]
    assert len(tables) == 1
    assert tables[0].title == "Local Branches"


def test_500_falls_back_to_local_branches(mocker: MockerFixture, repo_engine: Engine) -> None:
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature-1", repo.head.peel(pygit2.Commit))
    engine.state.gh = mocker.MagicMock()

    exc = GithubException(500, {"message": "Internal error"}, None)
    mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", side_effect=exc)
    engine.run_task(TaskType.COMMAND, "/review")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    texts = output_texts(drained_events)
    assert any("Falling back" in t for t in texts)
    tables = [e for e in drained_events if isinstance(e, TableOutput)]
    assert len(tables) == 1
    assert tables[0].title == "Local Branches"


# ── _truncate_output ─────────────────────────────────────────────────


def test_truncate_short_output_unchanged() -> None:

    text = "line1\nline2\nline3"
    result, hidden = _truncate_output(text, 5)
    assert result == text
    assert hidden == 0


def test_truncate_exact_limit_unchanged() -> None:

    text = "\n".join(f"line{i}" for i in range(25))
    result, hidden = _truncate_output(text, 25)
    assert result == text
    assert hidden == 0


def test_truncate_over_limit() -> None:

    text = "\n".join(f"line{i}" for i in range(30))
    result, hidden = _truncate_output(text, 25)
    assert result.count("\n") == 24  # 25 lines = 24 newlines
    assert hidden == 5


def test_truncate_empty_input() -> None:

    result, hidden = _truncate_output("", 25)
    assert result == ""
    assert hidden == 0


# ── Cancellation ─────────────────────────────────────────────────────


def test_cancel_shell_command(engine: Engine) -> None:
    """Cancelling a long-running shell command emits TaskFinished(cancelled=True)."""

    def run_and_cancel() -> None:
        import time

        time.sleep(0.1)
        engine.cancel()

    canceller = threading.Thread(target=run_and_cancel, daemon=True)
    canceller.start()
    engine.run_task(TaskType.SHELL, "sleep 30")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is False
    assert drained_events[-1].cancelled is True


def test_cancel_is_cleared_on_next_task(engine: Engine) -> None:
    """After cancellation, the next task runs normally."""

    def cancel_soon() -> None:
        import time

        time.sleep(0.1)
        engine.cancel()

    canceller = threading.Thread(target=cancel_soon, daemon=True)
    canceller.start()
    engine.run_task(TaskType.SHELL, "sleep 30")
    drain(engine.events)

    engine._cancel.clear()
    engine.run_task(TaskType.SHELL, "echo ok")
    drained_events = drain(engine.events)
    assert drained_events[-1].success is True
    assert drained_events[-1].cancelled is False
    texts = output_texts(drained_events)
    assert any("ok" in t for t in texts)


def test_cancel_noop_when_idle(engine: Engine) -> None:
    engine.cancel()  # should not raise


def test_cancel_check_raises(engine: Engine) -> None:

    engine._cancel.set()
    with pytest.raises(TaskCancelled):
        engine._check_cancel()


def test_cancel_does_not_lose_partial_output(engine: Engine) -> None:
    """Output emitted before cancellation is preserved in events."""

    def cancel_after_start() -> None:
        import time

        time.sleep(0.1)
        engine.cancel()

    canceller = threading.Thread(target=cancel_after_start, daemon=True)
    canceller.start()
    engine.run_task(TaskType.SHELL, "sleep 30")
    drained_events = drain(engine.events)
    texts = output_texts(drained_events)
    assert any("sleep 30" in t for t in texts)


# ── FlushPanel ───────────────────────────────────────────────────────


def test_list_clears_fetching_message(mocker: MockerFixture, engine: Engine) -> None:
    """The 'Fetching…' message is discarded before results appear."""
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", return_value=[PR_FIX_BUG])
    mocker.patch("rbtr.engine.review_cmd.client.list_unmerged_branches", return_value=[])
    engine.run_task(TaskType.COMMAND, "/review")

    drained_events = drain(engine.events)
    flush_events = [e for e in drained_events if isinstance(e, FlushPanel)]
    assert any(f.discard is True for f in flush_events)


def test_list_flushes_between_tables(mocker: MockerFixture, engine: Engine) -> None:
    """PR table and branch table are separated by a FlushPanel."""
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.list_open_prs", return_value=[PR_FIX_BUG])
    mocker.patch(
        "rbtr.engine.review_cmd.client.list_unmerged_branches", return_value=[BRANCH_FEATURE_X]
    )
    engine.run_task(TaskType.COMMAND, "/review")

    drained_events = drain(engine.events)
    tables_and_flushes = [
        e
        for e in drained_events
        if isinstance(e, TableOutput) or (isinstance(e, FlushPanel) and not e.discard)
    ]
    assert len(tables_and_flushes) == 3
    assert isinstance(tables_and_flushes[0], TableOutput)
    assert isinstance(tables_and_flushes[1], FlushPanel)
    assert isinstance(tables_and_flushes[2], TableOutput)


def test_review_pr_clears_fetching_message(mocker: MockerFixture, engine: Engine) -> None:
    """The 'Fetching PR #N…' message is discarded before result."""
    mock_gh = mocker.MagicMock()
    engine.state.gh = mock_gh

    mocker.patch("rbtr.engine.review_cmd.client.validate_pr_number", return_value=PR_ADD_FEATURE)
    engine.run_task(TaskType.COMMAND, "/review 42")

    drained_events = drain(engine.events)
    flush_events = [e for e in drained_events if isinstance(e, FlushPanel)]
    assert any(f.discard is True for f in flush_events)
    texts = output_texts(drained_events)
    assert any("PR #42" in t for t in texts)


def test_connect_github_flushes_link_panel(
    creds_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """/connect github emits link+code as one panel, then auth result as another."""

    device_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "device_code": "devcode",
        "interval": "0",
    }

    mocker.patch(
        "rbtr.engine.connect_cmd.github_auth.request_device_code", return_value=device_resp
    )
    mocker.patch("rbtr.engine.connect_cmd.github_auth.poll_for_token", return_value="ghp_tok")
    mocker.patch.object(engine, "_copy_to_clipboard")

    fake_gh = mocker.MagicMock()
    fake_gh.get_user.return_value.login = "testuser"
    mocker.patch("rbtr.engine.connect_cmd.Github", return_value=fake_gh)

    engine.run_task(TaskType.COMMAND, "/connect github")

    drained_events = drain(engine.events)
    assert drained_events[-1].success is True

    flush_events = [e for e in drained_events if isinstance(e, FlushPanel)]
    assert len(flush_events) == 2
    assert flush_events[0].discard is False  # link+code panel
    assert flush_events[1].discard is True  # "Waiting…" discarded

    assert has_event_type(drained_events, LinkOutput)
    texts = output_texts(drained_events)
    assert any("Authenticated" in t for t in texts)


# ── Usage tracking ───────────────────────────────────────────────────


def test_new_resets_usage(engine: Engine) -> None:
    """/new resets the state usage alongside message history."""
    engine.state.usage.record_run(
        input_tokens=5000,
        output_tokens=2000,
    )
    assert engine.state.usage.input_tokens == 5000

    engine.run_task(TaskType.COMMAND, "/new")
    drain(engine.events)
    assert engine.state.usage.input_tokens == 0
    assert engine.state.usage.output_tokens == 0
    assert engine.state.usage.total_cost == 0.0


def test_model_change_updates_state_model(config_path: Path, engine: Engine) -> None:
    """/model updates engine.state.model_name."""
    engine.state.cached_models = [("openai", ["openai/gpt-4o"])]

    engine.run_task(TaskType.COMMAND, "/model openai/gpt-4o")
    drain(engine.events)

    assert engine.state.model_name == "openai/gpt-4o"


def test_model_change_cross_provider_preserves_history(config_path: Path, engine: Engine) -> None:
    """Switching providers preserves message history in the DB."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine._sync_store_context()
    engine.store.save_messages(
        engine.state.session_id,
        [ModelRequest(parts=[UserPromptPart(content="keep me")])],
    )
    engine.state.usage.record_run(input_tokens=1000, output_tokens=500)
    engine.state.cached_models = [("chatgpt", ["chatgpt/gpt-4o"])]

    engine.run_task(TaskType.COMMAND, "/model chatgpt/gpt-4o")
    drain(engine.events)

    assert engine.state.model_name == "chatgpt/gpt-4o"
    assert len(engine.store.load_messages(engine.state.session_id)) == 1
    assert engine.state.usage.input_tokens == 1000


def test_model_change_same_provider_preserves_history(config_path: Path, engine: Engine) -> None:
    """Switching models within the same provider preserves history in the DB."""
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine._sync_store_context()
    engine.store.save_messages(
        engine.state.session_id,
        [ModelRequest(parts=[UserPromptPart(content="keep me")])],
    )
    engine.state.cached_models = [
        ("claude", ["claude/claude-sonnet-4-20250514", "claude/claude-opus-4-20250514"])
    ]

    engine.run_task(TaskType.COMMAND, "/model claude/claude-opus-4-20250514")
    drain(engine.events)

    assert engine.state.model_name == "claude/claude-opus-4-20250514"
    assert len(engine.store.load_messages(engine.state.session_id)) == 1


# ── EngineState.index & index events ─────────────────────────────────────


def test_state_index_defaults_to_none() -> None:
    state = EngineState()
    assert state.index is None


def test_new_preserves_index(mocker: MockerFixture, engine: Engine) -> None:
    """``/new`` clears conversation but leaves the index intact."""
    mock_store = mocker.MagicMock()
    engine.state.index = mock_store

    engine.run_task(TaskType.COMMAND, "/new")
    drain(engine.events)

    mock_store.close.assert_not_called()
    assert engine.state.index is mock_store


def test_index_events_are_valid() -> None:
    """Index event types are part of the Event union and serialize cleanly."""
    started = IndexStarted(total_files=100)
    progress = IndexProgress(phase="parsing", indexed=42, total=100)
    ready = IndexReady(chunk_count=350)

    assert started.total_files == 100
    assert progress.indexed == 42
    assert progress.total == 100
    assert ready.chunk_count == 350

    # Verify they're valid Event union members by putting them in a queue.
    eq: queue.Queue[Event] = queue.Queue()
    eq.put(started)
    eq.put(progress)
    eq.put(ready)
    assert eq.qsize() == 3


# ── Tool call events ─────────────────────────────────────────────────


def test_tool_call_events_serialize() -> None:
    """ToolCallStarted and ToolCallFinished are valid Event union members."""
    from rbtr.events import ToolCallFinished, ToolCallStarted

    started = ToolCallStarted(tool_name="search_symbols", args='{"name": "greet"}')
    finished = ToolCallFinished(tool_name="search_symbols", result="function greet (src/app.py:1)")

    assert started.tool_name == "search_symbols"
    assert finished.result.startswith("function greet")

    eq: queue.Queue[Event] = queue.Queue()
    eq.put(started)
    eq.put(finished)
    assert eq.qsize() == 2

    # Detailed _emit_tool_event tests live in test_llm.py (parametrised).
