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

from rbtr.constants import SHELL_MAX_LINES
from rbtr.engine import Engine, Session
from rbtr.events import (
    Event,
    FlushPanel,
    LinkOutput,
    MarkdownOutput,
    Output,
    TableOutput,
    TaskFinished,
    TaskStarted,
)
from rbtr.models import BranchSummary, PRSummary

# ── Helpers ──────────────────────────────────────────────────────────


def _make_engine(
    *,
    owner: str = "testowner",
    repo_name: str = "testrepo",
    gh: object | None = None,
    pr_number: int | None = None,
) -> tuple[Engine, queue.Queue[Event], Session]:
    """Create an Engine with a pre-populated Session."""
    session = Session(owner=owner, repo_name=repo_name, gh=gh)
    events: queue.Queue[Event] = queue.Queue()
    engine = Engine(session, events, pr_number=pr_number)
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
    engine.run_task("command", "/help")
    evts = _drain(events)

    assert isinstance(evts[0], TaskStarted)
    assert isinstance(evts[-1], TaskFinished)
    assert evts[-1].success is True

    texts = _output_texts(evts)
    commands_mentioned = [t for t in texts if "/help" in t or "/review" in t or "/quit" in t]
    assert len(commands_mentioned) >= 3


def test_unknown_command_warns() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("command", "/nonexistent")
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

    mocker.patch("rbtr.engine.client.list_open_prs", return_value=prs)
    mocker.patch("rbtr.engine.client.list_unmerged_branches", return_value=branches)
    engine.run_task("command", "/review")

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
        engine.run_task("command", "/review")

        evts = _drain(events)
        assert evts[-1].success is True
        tables = [e for e in evts if isinstance(e, TableOutput)]
        assert len(tables) == 1
        assert tables[0].title == "Local Branches"


def test_list_no_prs_no_branches(mocker) -> None:
    mock_gh = mocker.MagicMock()
    engine, events, _ = _make_engine(gh=mock_gh)

    mocker.patch("rbtr.engine.client.list_open_prs", return_value=[])
    mocker.patch("rbtr.engine.client.list_unmerged_branches", return_value=[])
    engine.run_task("command", "/review")

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

    mocker.patch("rbtr.engine.client.validate_pr_number", return_value=pr)
    engine.run_task("command", "/review 42")

    evts = _drain(events)
    assert evts[-1].success is True
    assert session.pr is not None
    assert session.pr.number == 42


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
        engine.run_task("command", "/review")
        evts = _drain(events)
        assert evts[-1].success is True
        assert _has_event_type(evts, TableOutput)


def test_review_pr_without_auth_warns() -> None:
    engine, events, _ = _make_engine(gh=None)
    engine.run_task("command", "/review 42")
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
        engine.run_task("command", "/review my-branch")

        evts = _drain(events)
        assert evts[-1].success is True
        assert session.pr is not None
        assert session.pr.head_branch == "my-branch"
        assert session.pr.number is None


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
        engine.run_task("command", "/review ghost-branch")

        evts = _drain(events)
        texts = _output_texts(evts)
        assert any("not found" in t for t in texts)


# ── !shell ───────────────────────────────────────────────────────────


def test_shell_captures_stdout() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("shell", "echo hello")
    evts = _drain(events)
    assert evts[-1].success is True
    texts = _output_texts(evts)
    assert any("hello" in t for t in texts)


def test_shell_captures_nonzero_exit() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("shell", "false")
    evts = _drain(events)
    assert evts[-1].success is True  # task itself succeeds; error is in output
    texts = _output_texts(evts)
    assert any("exit code" in t for t in texts)


def test_shell_empty_shows_usage() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("shell", "")
    evts = _drain(events)
    texts = _output_texts(evts)
    assert any("Usage" in t for t in texts)


def test_shell_truncates_long_output() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("shell", "seq 1 100")
    evts = _drain(events)
    _output_texts(evts)
    assert engine._last_shell_full_output is not None
    stdout_full, _, _, hidden_count = engine._last_shell_full_output
    assert len(stdout_full.split("\n")) > SHELL_MAX_LINES
    assert hidden_count > 0


def test_shell_never_emits_flush_panel() -> None:
    """Shell commands always produce a single panel — no FlushPanel."""
    engine, events, _ = _make_engine()
    engine.run_task("shell", "echo hello && echo world")
    evts = _drain(events)
    assert not any(isinstance(e, FlushPanel) for e in evts)


def test_shell_with_stderr_never_emits_flush_panel() -> None:
    """Even with mixed stdout/stderr, shell stays in one panel."""
    engine, events, _ = _make_engine()
    engine.run_task("shell", "echo out; echo err >&2")
    evts = _drain(events)
    assert not any(isinstance(e, FlushPanel) for e in evts)


# ── LLM placeholder ─────────────────────────────────────────────────


def test_llm_returns_placeholder() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("llm", "explain this code")
    evts = _drain(events)
    assert evts[-1].success is True
    assert _has_event_type(evts, MarkdownOutput)


# ── Setup ────────────────────────────────────────────────────────────


def test_setup_in_valid_repo(monkeypatch, mocker) -> None:
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

        mocker.patch("rbtr.engine.auth.load_token", return_value=None)
        engine.run_task("setup", "")

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

        engine.run_task("setup", "")

        evts = _drain(events)
        texts = _output_texts(evts)
        assert any("git repository" in t for t in texts)


# ── Event contract ───────────────────────────────────────────────────


def test_every_task_starts_and_finishes() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("command", "/help")
    evts = _drain(events)

    assert isinstance(evts[0], TaskStarted)
    assert isinstance(evts[-1], TaskFinished)


def test_task_finished_reports_success() -> None:
    engine, events, _ = _make_engine()
    engine.run_task("command", "/help")
    evts = _drain(events)
    assert evts[-1].success is True


def test_no_events_outside_task_boundary() -> None:
    """All Output events must be between TaskStarted and TaskFinished."""
    engine, events, _ = _make_engine()
    engine.run_task("command", "/review")
    evts = _drain(events)

    started = False
    for e in evts:
        if isinstance(e, TaskStarted):
            started = True
        elif isinstance(e, TaskFinished):
            started = False
        else:
            assert started, f"Event {type(e).__name__} emitted outside task boundary"


# ── /login ───────────────────────────────────────────────────────────


def test_login_success(mocker) -> None:
    engine, events, session = _make_engine(gh=None)

    device_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "device_code": "dev123",
        "interval": "0",
    }

    mocker.patch("rbtr.engine.auth.clear_token")
    mocker.patch("rbtr.engine.auth.request_device_code", return_value=device_resp)
    mocker.patch("rbtr.engine.auth.poll_for_token", return_value="ghp_newtoken")
    mocker.patch("rbtr.engine.auth.save_token")
    mocker.patch.object(Engine, "_copy_to_clipboard")
    engine.run_task("command", "/login")

    evts = _drain(events)
    assert evts[-1].success is True
    assert session.gh is not None

    links = [e for e in evts if isinstance(e, LinkOutput)]
    assert len(links) == 1
    assert "ABCD-1234" in links[0].markup
    assert "github.com/login/device" in links[0].markup

    texts = _output_texts(evts)
    assert any("Authenticated" in t for t in texts)


def test_login_failure(mocker) -> None:
    from rbtr import RbtrError

    engine, events, session = _make_engine(gh=None)

    mocker.patch("rbtr.engine.auth.clear_token")
    mocker.patch(
        "rbtr.engine.auth.request_device_code",
        side_effect=RbtrError("network error"),
    )
    engine.run_task("command", "/login")

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
        mocker.patch("rbtr.engine.client.list_open_prs", side_effect=exc)
        engine.run_task("command", "/review")

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
        mocker.patch("rbtr.engine.client.list_open_prs", side_effect=exc)
        engine.run_task("command", "/review")

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
    result, hidden = Engine._truncate_output(text, 5)
    assert result == text
    assert hidden == 0


def test_truncate_exact_limit_unchanged() -> None:
    text = "\n".join(f"line{i}" for i in range(25))
    result, hidden = Engine._truncate_output(text, 25)
    assert result == text
    assert hidden == 0


def test_truncate_over_limit() -> None:
    text = "\n".join(f"line{i}" for i in range(30))
    result, hidden = Engine._truncate_output(text, 25)
    assert result.count("\n") == 24  # 25 lines = 24 newlines
    assert hidden == 5


def test_truncate_empty_input() -> None:
    result, hidden = Engine._truncate_output("", 25)
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
    engine.run_task("shell", "sleep 30")
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
    engine.run_task("shell", "sleep 30")
    _drain(events)

    engine._cancel.clear()
    engine.run_task("shell", "echo ok")
    evts = _drain(events)
    assert evts[-1].success is True
    assert evts[-1].cancelled is False
    texts = _output_texts(evts)
    assert any("ok" in t for t in texts)


def test_cancel_noop_when_idle() -> None:
    engine, _events, _ = _make_engine()
    engine.cancel()  # should not raise


def test_cancel_check_raises() -> None:
    from rbtr.engine import _TaskCancelled

    engine, _, _ = _make_engine()
    engine._cancel.set()
    with pytest.raises(_TaskCancelled):
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
    engine.run_task("shell", "sleep 30")
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

    mocker.patch("rbtr.engine.client.list_open_prs", return_value=prs)
    mocker.patch("rbtr.engine.client.list_unmerged_branches", return_value=[])
    engine.run_task("command", "/review")

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

    mocker.patch("rbtr.engine.client.list_open_prs", return_value=prs)
    mocker.patch("rbtr.engine.client.list_unmerged_branches", return_value=branches)
    engine.run_task("command", "/review")

    evts = _drain(events)
    tables_and_flushes = [
        e
        for e in evts
        if isinstance(e, TableOutput)
        or (isinstance(e, FlushPanel) and not e.discard)
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

    mocker.patch("rbtr.engine.client.validate_pr_number", return_value=pr)
    engine.run_task("command", "/review 42")

    evts = _drain(events)
    flush_events = [e for e in evts if isinstance(e, FlushPanel)]
    assert any(f.discard is True for f in flush_events)
    texts = _output_texts(evts)
    assert any("PR #42" in t for t in texts)


def test_login_flushes_link_panel(mocker) -> None:
    """Login emits link+code as one panel, then auth result as another."""
    engine, events, _ = _make_engine()

    device_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "device_code": "devcode",
        "interval": "0",
    }

    mocker.patch("rbtr.engine.auth.clear_token")
    mocker.patch("rbtr.engine.auth.request_device_code", return_value=device_resp)
    mocker.patch("rbtr.engine.auth.poll_for_token", return_value="ghp_tok")
    mocker.patch("rbtr.engine.auth.save_token")
    mocker.patch.object(engine, "_copy_to_clipboard")
    engine.run_task("command", "/login")

    evts = _drain(events)
    assert evts[-1].success is True

    flush_events = [e for e in evts if isinstance(e, FlushPanel)]
    assert len(flush_events) == 2
    assert flush_events[0].discard is False  # link+code panel
    assert flush_events[1].discard is True   # "Waiting…" discarded

    assert _has_event_type(evts, LinkOutput)
    texts = _output_texts(evts)
    assert any("Authenticated" in t for t in texts)
