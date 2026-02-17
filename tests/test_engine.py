"""End-to-end tests for the Engine via its event contract.

These tests create a real Engine with a mock Session, run tasks
synchronously (no daemon threads needed — run_task is a plain method),
and assert on the sequence of events emitted to the queue.
"""

import queue
import tempfile
import threading
from datetime import UTC, datetime

import pygit2
import pytest
from github.GithubException import GithubException
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)

from rbtr import RbtrError
from rbtr.config import config
from rbtr.creds import OAuthCreds, creds
from rbtr.engine import Engine, Session, TaskCancelled, TaskType
from rbtr.engine.history import demote_thinking, is_history_format_error
from rbtr.engine.shell import _truncate_output
from rbtr.events import (
    Event,
    FlushPanel,
    LinkOutput,
    Output,
    TableOutput,
    TaskFinished,
    TaskStarted,
    TextDelta,
)
from rbtr.models import BranchSummary, BranchTarget, PRSummary, PRTarget

# ── Helpers ──────────────────────────────────────────────────────────


def _make_engine(
    *,
    owner: str = "testowner",
    repo_name: str = "testrepo",
    gh: object | None = None,
) -> tuple[Engine, queue.Queue[Event], Session]:
    """Create an Engine with a pre-populated Session."""
    session = Session(owner=owner, repo_name=repo_name, gh=gh)
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(session, events)
    return engine, events, session


def _drain(events: queue.Queue[Event]) -> list[Event]:
    """Drain all events from the queue into a list."""
    result: list[Event] = []
    while True:
        try:
            result.append(events.get_nowait())
        except queue.Empty:
            break
    return result


def _output_texts(events: list[Event]) -> list[str]:
    """Extract text from Output events."""
    return [e.text for e in events if isinstance(e, Output)]


def _has_event_type(events: list[Event], event_type: type) -> bool:
    return any(isinstance(e, event_type) for e in events)


# ── /help ────────────────────────────────────────────────────────────


def test_help_lists_all_commands() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.COMMAND, "/help")
    evts = _drain(events)

    assert isinstance(evts[0], TaskStarted)
    assert isinstance(evts[-1], TaskFinished)
    assert evts[-1].success is True

    texts = _output_texts(evts)
    commands_mentioned = [t for t in texts if "/help" in t or "/review" in t or "/quit" in t]
    assert len(commands_mentioned) >= 3


def test_unknown_command_warns() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.COMMAND, "/nonexistent")
    evts = _drain(events)

    texts = _output_texts(evts)
    assert any("Unknown command" in t for t in texts)
    assert evts[-1].success is True


# ── /review (list mode) ─────────────────────────────────────────────


def test_list_with_github_prs(mocker) -> None:
    mock_gh = mocker.MagicMock()
    engine, events, _ = _make_engine(gh=mock_gh)

    prs = [
        PRSummary(
            number=1,
            title="Fix bug",
            author="alice",
            head_branch="fix-bug",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    ]
    branches = [
        BranchSummary(
            name="feature-x",
            last_commit_sha="abc123",
            last_commit_message="wip",
            updated_at=datetime(2025, 1, 2, tzinfo=UTC),
        ),
    ]

    mocker.patch("rbtr.engine.review.client.list_open_prs", return_value=prs)
    mocker.patch("rbtr.engine.review.client.list_unmerged_branches", return_value=branches)
    engine.run_task(TaskType.COMMAND, "/review")

    evts = _drain(events)
    assert evts[-1].success is True
    assert _has_event_type(evts, TableOutput)

    tables = [e for e in evts if isinstance(e, TableOutput)]
    assert len(tables) == 2
    assert tables[0].title == "Open Pull Requests"
    assert tables[1].title == "Unmerged Branches"


def test_list_without_auth_falls_back_to_local() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("Test", "test@test.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.branches.local.create("feature-1", repo.head.peel(pygit2.Commit))

        session = Session(repo=repo, owner="o", repo_name="r", gh=None)
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)
        engine.run_task(TaskType.COMMAND, "/review")

        evts = _drain(events)
        assert evts[-1].success is True
        tables = [e for e in evts if isinstance(e, TableOutput)]
        assert len(tables) == 1
        assert tables[0].title == "Local Branches"


def test_list_no_prs_no_branches(mocker) -> None:
    mock_gh = mocker.MagicMock()
    engine, events, _ = _make_engine(gh=mock_gh)

    mocker.patch("rbtr.engine.review.client.list_open_prs", return_value=[])
    mocker.patch("rbtr.engine.review.client.list_unmerged_branches", return_value=[])
    engine.run_task(TaskType.COMMAND, "/review")

    evts = _drain(events)
    texts = _output_texts(evts)
    assert any("No open PRs" in t for t in texts)


# ── /review <target> ────────────────────────────────────────────────


def test_review_pr_by_number(mocker) -> None:
    mock_gh = mocker.MagicMock()
    engine, events, session = _make_engine(gh=mock_gh)

    pr = PRSummary(
        number=42,
        title="Add feature",
        author="bob",
        head_branch="add-feature",
        updated_at=datetime(2025, 6, 1, tzinfo=UTC),
    )

    mocker.patch("rbtr.engine.review.client.validate_pr_number", return_value=pr)
    engine.run_task(TaskType.COMMAND, "/review 42")

    evts = _drain(events)
    assert evts[-1].success is True
    assert isinstance(session.review_target, PRTarget)
    assert session.review_target.number == 42


def test_review_without_arg_lists() -> None:
    """/review with no argument behaves like the old /list."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("Test", "test@test.com")
        tree = repo.TreeBuilder().write()
        oid = repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.create_branch("feature-x", repo.get(oid))

        engine, events, session = _make_engine(gh=None)
        session.repo = repo
        engine.run_task(TaskType.COMMAND, "/review")
        evts = _drain(events)
        assert evts[-1].success is True
        assert _has_event_type(evts, TableOutput)


def test_review_pr_without_auth_warns() -> None:
    engine, events, _ = _make_engine(gh=None)
    engine.run_task(TaskType.COMMAND, "/review 42")
    evts = _drain(events)
    texts = _output_texts(evts)
    assert any("Not authenticated" in t for t in texts)


def test_review_branch_by_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("Test", "test@test.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.branches.local.create("my-branch", repo.head.peel(pygit2.Commit))

        session = Session(repo=repo, owner="o", repo_name="r", gh=None)
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)
        engine.run_task(TaskType.COMMAND, "/review my-branch")

        evts = _drain(events)
        assert evts[-1].success is True
        assert isinstance(session.review_target, BranchTarget)
        assert session.review_target.head_branch == "my-branch"


def test_review_nonexistent_branch_warns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("Test", "test@test.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")

        session = Session(repo=repo, owner="o", repo_name="r", gh=None)
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)
        engine.run_task(TaskType.COMMAND, "/review ghost-branch")

        evts = _drain(events)
        texts = _output_texts(evts)
        assert any("not found" in t for t in texts)


# ── !shell ───────────────────────────────────────────────────────────


def test_shell_captures_stdout() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.SHELL, "echo hello")
    evts = _drain(events)
    assert evts[-1].success is True
    texts = _output_texts(evts)
    assert any("hello" in t for t in texts)


def test_shell_captures_nonzero_exit() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.SHELL, "false")
    evts = _drain(events)
    assert evts[-1].success is True  # task itself succeeds; error is in output
    texts = _output_texts(evts)
    assert any("exit code" in t for t in texts)


def test_shell_empty_shows_usage() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.SHELL, "")
    evts = _drain(events)
    texts = _output_texts(evts)
    assert any("Usage" in t for t in texts)


def test_shell_truncates_long_output() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.SHELL, "seq 1 100")
    evts = _drain(events)
    _output_texts(evts)
    assert engine._last_shell_full_output is not None
    stdout_full, _, _, hidden_count = engine._last_shell_full_output
    assert len(stdout_full.split("\n")) > config.tui.shell_max_lines
    assert hidden_count > 0


def test_shell_never_emits_flush_panel() -> None:
    """Shell commands always produce a single panel — no FlushPanel."""
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.SHELL, "echo hello && echo world")
    evts = _drain(events)
    assert not any(isinstance(e, FlushPanel) for e in evts)


def test_shell_with_stderr_never_emits_flush_panel() -> None:
    """Even with mixed stdout/stderr, shell stays in one panel."""
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.SHELL, "echo out; echo err >&2")
    evts = _drain(events)
    assert not any(isinstance(e, FlushPanel) for e in evts)


# ── LLM placeholder ─────────────────────────────────────────────────


def test_llm_streams_response(creds_path, mocker) -> None:
    """LLM messages build a model and stream via the agent."""

    creds.update(claude=OAuthCreds(access_token="t", refresh_token="r", expires_at=9e9))

    engine, events, session = _make_engine()
    session.claude_connected = True

    async def fake_stream(eng, model, message):
        from rbtr.events import TextDelta

        eng._emit(TextDelta(delta="Hello "))
        eng._emit(TextDelta(delta="world"))

    mocker.patch("rbtr.engine.llm._stream_agent", fake_stream)
    engine.run_task(TaskType.LLM, "explain this code")
    evts = _drain(events)
    assert evts[-1].success is True
    deltas = [e for e in evts if isinstance(e, TextDelta)]
    assert len(deltas) == 2
    assert deltas[0].delta == "Hello "
    assert deltas[1].delta == "world"


def test_llm_preserves_history(creds_path, mocker) -> None:
    """After an LLM call, message_history is populated."""

    creds.update(openai_api_key="sk-test")

    engine, events, session = _make_engine()
    session.openai_connected = True

    async def fake_stream(eng, model, message):
        eng.session.message_history = [{"role": "user", "content": message}]

    mocker.patch("rbtr.engine.llm._stream_agent", fake_stream)
    engine.run_task(TaskType.LLM, "hello")
    _drain(events)
    assert len(session.message_history) == 1


def test_new_clears_history() -> None:
    engine, events, session = _make_engine()
    session.message_history = [{"role": "user", "content": "old"}]
    engine.run_task(TaskType.COMMAND, "/new")
    evts = _drain(events)
    assert evts[-1].success is True
    assert session.message_history == []
    texts = _output_texts(evts)
    assert any("cleared" in t.lower() for t in texts)


def test_llm_warns_when_not_connected() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.LLM, "explain this code")
    evts = _drain(events)
    assert evts[-1].success is True
    warnings = [e for e in evts if isinstance(e, Output) and "No LLM connected" in e.text]
    assert len(warnings) == 1


# ── /connect openai ──────────────────────────────────────────────────


def test_connect_openai_saves_key(creds_path) -> None:

    engine, events, session = _make_engine()
    engine.run_task(TaskType.COMMAND, "/connect openai sk-test-key-123")
    evts = _drain(events)
    assert evts[-1].success is True
    assert session.openai_connected is True
    assert creds.openai_api_key == "sk-test-key-123"
    texts = _output_texts(evts)
    assert any("Connected to OpenAI" in t for t in texts)


def test_connect_openai_already_connected(creds_path) -> None:

    creds.update(openai_api_key="sk-existing")
    engine, events, session = _make_engine()
    engine.run_task(TaskType.COMMAND, "/connect openai")
    evts = _drain(events)
    assert evts[-1].success is True
    assert session.openai_connected is True
    texts = _output_texts(evts)
    assert any("Already connected" in t for t in texts)


def test_connect_openai_rejects_bad_key(creds_path) -> None:
    engine, events, session = _make_engine()
    engine.run_task(TaskType.COMMAND, "/connect openai bad-key-format")
    evts = _drain(events)
    assert evts[-1].success is True
    assert session.openai_connected is False
    texts = _output_texts(evts)
    assert any("Invalid" in t for t in texts)


def test_connect_openai_no_key_shows_usage(creds_path) -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.COMMAND, "/connect openai")
    evts = _drain(events)
    texts = _output_texts(evts)
    assert any("Usage" in t for t in texts)
    assert any("platform.openai.com" in t for t in texts)


def test_connect_openai_replaces_existing_key(creds_path) -> None:
    """Providing a key when one exists replaces it."""

    creds.update(openai_api_key="sk-old")
    engine, events, session = _make_engine()
    engine.run_task(TaskType.COMMAND, "/connect openai sk-new-key")
    evts = _drain(events)
    assert evts[-1].success is True
    assert session.openai_connected is True
    assert creds.openai_api_key == "sk-new-key"


# ── Setup ────────────────────────────────────────────────────────────


def test_setup_in_valid_repo(monkeypatch, creds_path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("Test", "test@test.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.remotes.create("origin", "git@github.com:testowner/testrepo.git")

        monkeypatch.chdir(tmp)
        session = Session()
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")

        evts = _drain(events)
        assert evts[-1].success is True
        assert session.owner == "testowner"
        assert session.repo_name == "testrepo"
        texts = _output_texts(evts)
        assert any("testowner/testrepo" in t for t in texts)
        assert any("Not authenticated" in t for t in texts)


def test_setup_outside_repo_errors(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        session = Session()
        events: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events)

        engine.run_task(TaskType.SETUP, "")

        evts = _drain(events)
        texts = _output_texts(evts)
        assert any("git repository" in t for t in texts)


# ── Event contract ───────────────────────────────────────────────────


def test_every_task_starts_and_finishes() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.COMMAND, "/help")
    evts = _drain(events)

    assert isinstance(evts[0], TaskStarted)
    assert isinstance(evts[-1], TaskFinished)


def test_task_finished_reports_success() -> None:
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.COMMAND, "/help")
    evts = _drain(events)
    assert evts[-1].success is True


def test_no_events_outside_task_boundary() -> None:
    """All Output events must be between TaskStarted and TaskFinished."""
    engine, events, _ = _make_engine()
    engine.run_task(TaskType.COMMAND, "/review")
    evts = _drain(events)

    started = False
    for e in evts:
        if isinstance(e, TaskStarted):
            started = True
        elif isinstance(e, TaskFinished):
            started = False
        else:
            assert started, f"Event {type(e).__name__} emitted outside task boundary"


# ── /connect github ───────────────────────────────────────────────────


def test_connect_github_success(creds_path, mocker) -> None:

    engine, events, session = _make_engine(gh=None)

    device_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "device_code": "dev123",
        "interval": "0",
    }

    mocker.patch("rbtr.engine.connect.auth.request_device_code", return_value=device_resp)
    mocker.patch("rbtr.engine.connect.auth.poll_for_token", return_value="ghp_newtoken")
    mocker.patch.object(Engine, "_copy_to_clipboard")
    engine.run_task(TaskType.COMMAND, "/connect github")

    evts = _drain(events)
    assert evts[-1].success is True
    assert session.gh is not None
    assert creds.github_token == "ghp_newtoken"

    links = [e for e in evts if isinstance(e, LinkOutput)]
    assert len(links) == 1
    assert "ABCD-1234" in links[0].markup
    assert "github.com/login/device" in links[0].markup

    texts = _output_texts(evts)
    assert any("Authenticated" in t for t in texts)


def test_connect_github_failure(creds_path, mocker) -> None:
    engine, events, session = _make_engine(gh=None)

    mocker.patch(
        "rbtr.engine.connect.auth.request_device_code",
        side_effect=RbtrError("network error"),
    )
    engine.run_task(TaskType.COMMAND, "/connect github")

    evts = _drain(events)
    assert evts[-1].success is True
    assert session.gh is None
    texts = _output_texts(evts)
    assert any("failed" in t.lower() for t in texts)


# ── GitHub error fallbacks ───────────────────────────────────────────


def _make_repo_with_branch(tmp: str) -> pygit2.Repository:
    """Create a repo with main + feature-1 branch."""
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")
    tree = repo.TreeBuilder().write()
    repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
    repo.set_head("refs/heads/main")
    repo.branches.local.create("feature-1", repo.head.peel(pygit2.Commit))
    return repo


def test_403_falls_back_to_local_branches(mocker) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_with_branch(tmp)
        mock_gh = mocker.MagicMock()
        session = Session(repo=repo, owner="o", repo_name="r", gh=mock_gh)
        events_q: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events_q)

        exc = GithubException(403, {"message": "Resource not accessible"}, None)
        mocker.patch("rbtr.engine.review.client.list_open_prs", side_effect=exc)
        engine.run_task(TaskType.COMMAND, "/review")

        evts = _drain(events_q)
        assert evts[-1].success is True
        texts = _output_texts(evts)
        assert any("Cannot access" in t for t in texts)
        tables = [e for e in evts if isinstance(e, TableOutput)]
        assert len(tables) == 1
        assert tables[0].title == "Local Branches"


def test_500_falls_back_to_local_branches(mocker) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo_with_branch(tmp)
        mock_gh = mocker.MagicMock()
        session = Session(repo=repo, owner="o", repo_name="r", gh=mock_gh)
        events_q: queue.Queue[Event] = queue.Queue()
        engine = Engine(session, events_q)

        exc = GithubException(500, {"message": "Internal error"}, None)
        mocker.patch("rbtr.engine.review.client.list_open_prs", side_effect=exc)
        engine.run_task(TaskType.COMMAND, "/review")

        evts = _drain(events_q)
        assert evts[-1].success is True
        texts = _output_texts(evts)
        assert any("Falling back" in t for t in texts)
        tables = [e for e in evts if isinstance(e, TableOutput)]
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


def test_cancel_shell_command() -> None:
    """Cancelling a long-running shell command emits TaskFinished(cancelled=True)."""
    engine, events, _ = _make_engine()

    def run_and_cancel() -> None:
        import time

        time.sleep(0.1)
        engine.cancel()

    canceller = threading.Thread(target=run_and_cancel, daemon=True)
    canceller.start()
    engine.run_task(TaskType.SHELL, "sleep 30")
    evts = _drain(events)
    assert evts[-1].success is False
    assert evts[-1].cancelled is True


def test_cancel_is_cleared_on_next_task() -> None:
    """After cancellation, the next task runs normally."""
    engine, events, _ = _make_engine()

    def cancel_soon() -> None:
        import time

        time.sleep(0.1)
        engine.cancel()

    canceller = threading.Thread(target=cancel_soon, daemon=True)
    canceller.start()
    engine.run_task(TaskType.SHELL, "sleep 30")
    _drain(events)

    engine._cancel.clear()
    engine.run_task(TaskType.SHELL, "echo ok")
    evts = _drain(events)
    assert evts[-1].success is True
    assert evts[-1].cancelled is False
    texts = _output_texts(evts)
    assert any("ok" in t for t in texts)


def test_cancel_noop_when_idle() -> None:
    engine, _events, _ = _make_engine()
    engine.cancel()  # should not raise


def test_cancel_check_raises() -> None:

    engine, _, _ = _make_engine()
    engine._cancel.set()
    with pytest.raises(TaskCancelled):
        engine._check_cancel()


def test_cancel_does_not_lose_partial_output() -> None:
    """Output emitted before cancellation is preserved in events."""
    engine, events, _ = _make_engine()

    def cancel_after_start() -> None:
        import time

        time.sleep(0.1)
        engine.cancel()

    canceller = threading.Thread(target=cancel_after_start, daemon=True)
    canceller.start()
    engine.run_task(TaskType.SHELL, "sleep 30")
    evts = _drain(events)
    texts = _output_texts(evts)
    assert any("sleep 30" in t for t in texts)


# ── FlushPanel ───────────────────────────────────────────────────────


def test_list_clears_fetching_message(mocker) -> None:
    """The 'Fetching…' message is discarded before results appear."""
    mock_gh = mocker.MagicMock()
    engine, events, _ = _make_engine(gh=mock_gh)

    prs = [
        PRSummary(
            number=1,
            title="Fix",
            author="a",
            head_branch="fix",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    ]

    mocker.patch("rbtr.engine.review.client.list_open_prs", return_value=prs)
    mocker.patch("rbtr.engine.review.client.list_unmerged_branches", return_value=[])
    engine.run_task(TaskType.COMMAND, "/review")

    evts = _drain(events)
    flush_events = [e for e in evts if isinstance(e, FlushPanel)]
    assert any(f.discard is True for f in flush_events)


def test_list_flushes_between_tables(mocker) -> None:
    """PR table and branch table are separated by a FlushPanel."""
    mock_gh = mocker.MagicMock()
    engine, events, _ = _make_engine(gh=mock_gh)

    prs = [
        PRSummary(
            number=1,
            title="Fix",
            author="a",
            head_branch="fix",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    ]
    branches = [
        BranchSummary(
            name="feat",
            last_commit_sha="abc",
            last_commit_message="wip",
            updated_at=datetime(2025, 1, 2, tzinfo=UTC),
        ),
    ]

    mocker.patch("rbtr.engine.review.client.list_open_prs", return_value=prs)
    mocker.patch("rbtr.engine.review.client.list_unmerged_branches", return_value=branches)
    engine.run_task(TaskType.COMMAND, "/review")

    evts = _drain(events)
    tables_and_flushes = [
        e
        for e in evts
        if isinstance(e, TableOutput) or (isinstance(e, FlushPanel) and not e.discard)
    ]
    assert len(tables_and_flushes) == 3
    assert isinstance(tables_and_flushes[0], TableOutput)
    assert isinstance(tables_and_flushes[1], FlushPanel)
    assert isinstance(tables_and_flushes[2], TableOutput)


def test_review_pr_clears_fetching_message(mocker) -> None:
    """The 'Fetching PR #N…' message is discarded before result."""
    mock_gh = mocker.MagicMock()
    engine, events, _session = _make_engine(gh=mock_gh)

    pr = PRSummary(
        number=42,
        title="Add feature",
        author="bob",
        head_branch="add-feature",
        updated_at=datetime(2025, 6, 1, tzinfo=UTC),
    )

    mocker.patch("rbtr.engine.review.client.validate_pr_number", return_value=pr)
    engine.run_task(TaskType.COMMAND, "/review 42")

    evts = _drain(events)
    flush_events = [e for e in evts if isinstance(e, FlushPanel)]
    assert any(f.discard is True for f in flush_events)
    texts = _output_texts(evts)
    assert any("PR #42" in t for t in texts)


def test_connect_github_flushes_link_panel(creds_path, mocker) -> None:
    """/connect github emits link+code as one panel, then auth result as another."""
    engine, events, _ = _make_engine()

    device_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "device_code": "devcode",
        "interval": "0",
    }

    mocker.patch("rbtr.engine.connect.auth.request_device_code", return_value=device_resp)
    mocker.patch("rbtr.engine.connect.auth.poll_for_token", return_value="ghp_tok")
    mocker.patch.object(engine, "_copy_to_clipboard")
    engine.run_task(TaskType.COMMAND, "/connect github")

    evts = _drain(events)
    assert evts[-1].success is True

    flush_events = [e for e in evts if isinstance(e, FlushPanel)]
    assert len(flush_events) == 2
    assert flush_events[0].discard is False  # link+code panel
    assert flush_events[1].discard is True  # "Waiting…" discarded

    assert _has_event_type(evts, LinkOutput)
    texts = _output_texts(evts)
    assert any("Authenticated" in t for t in texts)


# ── Usage tracking ───────────────────────────────────────────────────


def test_new_resets_usage() -> None:
    """/new resets the session usage alongside message history."""
    engine, events, session = _make_engine()
    session.usage.record_run(
        input_tokens=5000,
        output_tokens=2000,
    )
    assert session.usage.input_tokens == 5000

    engine.run_task(TaskType.COMMAND, "/new")
    _drain(events)
    assert session.usage.input_tokens == 0
    assert session.usage.output_tokens == 0
    assert session.usage.total_cost == 0.0


def test_model_change_updates_session_model(config_path) -> None:
    """/model updates session.model_name."""
    engine, events, session = _make_engine()
    session.cached_models = [("openai", ["openai/gpt-4o"])]

    engine.run_task(TaskType.COMMAND, "/model openai/gpt-4o")
    _drain(events)

    assert session.model_name == "openai/gpt-4o"


def test_model_change_cross_provider_preserves_history(config_path) -> None:
    """Switching providers preserves message history."""
    engine, events, session = _make_engine()
    session.model_name = "claude/claude-sonnet-4-20250514"
    session.message_history = [{"role": "user", "content": "keep me"}]
    session.usage.record_run(input_tokens=1000, output_tokens=500)
    session.cached_models = [("chatgpt", ["chatgpt/gpt-4o"])]

    engine.run_task(TaskType.COMMAND, "/model chatgpt/gpt-4o")
    _drain(events)

    assert session.model_name == "chatgpt/gpt-4o"
    assert len(session.message_history) == 1
    assert session.usage.input_tokens == 1000


def test_model_change_same_provider_preserves_history(config_path) -> None:
    """Switching models within the same provider preserves history."""
    engine, events, session = _make_engine()
    session.model_name = "claude/claude-sonnet-4-20250514"
    session.message_history = [{"role": "user", "content": "keep me"}]
    session.cached_models = [
        ("claude", ["claude/claude-sonnet-4-20250514", "claude/claude-opus-4-20250514"])
    ]

    engine.run_task(TaskType.COMMAND, "/model claude/claude-opus-4-20250514")
    _drain(events)

    assert session.model_name == "claude/claude-opus-4-20250514"
    assert len(session.message_history) == 1


# ── History repair ───────────────────────────────────────────────────


def test_demote_thinking_converts_to_text() -> None:

    history: list[object] = [
        ModelResponse(
            parts=[
                ThinkingPart(content="reasoning…", id="reasoning_content"),
                TextPart(content="hello"),
            ],
            model_name="test",
        ),
    ]

    cleaned = demote_thinking(history)
    assert len(cleaned) == 1
    response = cleaned[0]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 2
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == "<thinking>\nreasoning…\n</thinking>"
    assert isinstance(response.parts[1], TextPart)
    assert response.parts[1].content == "hello"


def test_demote_thinking_drops_empty_thinking() -> None:

    history: list[object] = [
        ModelResponse(
            parts=[ThinkingPart(content="", id="rs_123")],
            model_name="test",
        ),
    ]

    cleaned = demote_thinking(history)
    assert len(cleaned) == 0


def test_demote_thinking_preserves_non_responses() -> None:

    history: list[object] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
    ]

    cleaned = demote_thinking(history)
    assert len(cleaned) == 1


def test_is_history_format_error_invalid_id() -> None:

    exc = ModelHTTPError(
        400,
        "gpt-5.1-codex",
        body={
            "message": "Invalid 'input[6].id': 'reasoning_content'. "
            "Expected an ID that begins with 'rs'.",
        },
    )
    assert is_history_format_error(exc)


def test_is_history_format_error_missing_reasoning() -> None:

    exc = ModelHTTPError(
        400,
        "gpt-5-mini",
        body={
            "message": "Item 'fc_07f6' of type 'function_call' was "
            "provided without its required 'reasoning' item: 'rs_07f6'.",
        },
    )
    assert is_history_format_error(exc)


def test_is_history_format_error_rejects_unrelated() -> None:

    exc = ModelHTTPError(
        400,
        "gpt-4o",
        body={"message": "maximum context length exceeded"},
    )
    assert not is_history_format_error(exc)
