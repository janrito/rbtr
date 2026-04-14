"""Lifecycle tests for /draft — show, sync, post, clear.

All tests drive through `cmd_draft(draft_engine, ...)` with an
autospecced PyGithub mock.  No Fake* classes, no EngineStub.
"""

from __future__ import annotations

import pytest
from github.PullRequest import PullRequest
from pytest_mock import MockerFixture

from rbtr.engine.core import Engine
from rbtr.engine.draft_cmd import cmd_draft
from rbtr.events import MarkdownOutput
from rbtr.exceptions import RbtrError
from rbtr.github.draft import _comment_hash, load_draft, save_draft, stamp_synced
from rbtr.models import DiffSide, InlineComment, PRTarget, ReviewDraft
from tests.helpers import drain, output_texts

from .conftest import MockPRComment, MockReview, make_comment

# ── Show (/draft) ───────────────────────────────────────────────────


def test_show_no_draft(draft_engine: Engine) -> None:
    cmd_draft(draft_engine, "")
    assert "No draft" in "\n".join(output_texts(drain(draft_engine.events)))


def test_show_draft_with_comments(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="Looks good with minor issues.",
            comments=[
                InlineComment(
                    path="src/client.py", line=42, body="**blocker:** Retry without backoff."
                ),
                InlineComment(path="src/config.py", line=8, body="**nit:** Unused import."),
            ],
        ),
    )
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "## Summary" in text
    assert "Looks good" in text
    assert "2 comments" in text
    assert "### src/client.py" in text
    assert "### src/config.py" in text


def test_show_summary_at_bottom(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="Overview.",
            comments=[InlineComment(path="a.py", line=1, body="Fix.")],
        ),
    )
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert text.find("1 comment") < text.find("## Summary")


def test_show_empty_summary(draft_engine: Engine) -> None:
    save_draft(42, ReviewDraft(summary="", comments=[]))
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "(empty)" in text
    assert "No inline comments" in text


def test_show_no_comments(draft_engine: Engine) -> None:
    save_draft(42, ReviewDraft(summary="Just a summary."))
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "No inline comments" in text
    assert "Just a summary." in text


def test_show_empty_summary_label(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(comments=[InlineComment(path="a.py", line=1, body="Comment.")]),
    )
    cmd_draft(draft_engine, "")
    assert "(empty)" in "\n".join(output_texts(drain(draft_engine.events)))


def test_show_suggestion_as_code_block(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="",
            comments=[
                InlineComment(
                    path="src/client.py",
                    line=42,
                    body="Use exponential backoff.",
                    suggestion="time.sleep(2 ** attempt)",
                ),
            ],
        ),
    )
    cmd_draft(draft_engine, "")
    md = [ev for ev in drain(draft_engine.events) if isinstance(ev, MarkdownOutput)]
    md_text = "\n".join(ev.text for ev in md)
    assert "```suggestion" in md_text
    assert "time.sleep(2 ** attempt)" in md_text


def test_show_markdown_body_emitted(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="Summary text.",
            comments=[InlineComment(path="a.py", line=1, body="**bold** and `code`.")],
        ),
    )
    cmd_draft(draft_engine, "")
    md = [ev.text for ev in drain(draft_engine.events) if isinstance(ev, MarkdownOutput)]
    assert any("**bold** and `code`." in t for t in md)
    assert any("## Summary" in t for t in md)
    assert any("Summary text." in t for t in md)


def test_show_comments_grouped_by_file(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="Overview.",
            comments=[
                InlineComment(path="src/api.py", line=10, body="First comment."),
                InlineComment(path="src/db.py", line=5, body="DB comment."),
                InlineComment(path="src/api.py", line=20, body="Second comment on same file."),
            ],
        ),
    )
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "### src/api.py" in text
    assert "### src/db.py" in text
    api_pos = text.find("### src/api.py")
    db_pos = text.find("### src/db.py")
    first_pos = text.find("First comment.")
    second_pos = text.find("Second comment on same file.")
    assert api_pos < first_pos < second_pos
    assert api_pos < db_pos


def test_show_tombstone_marker(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[
                InlineComment(path="a.py", line=5, body="", github_id=100, comment_hash="abc"),
            ],
        ),
    )
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "deleted" in text.lower()
    assert "✗" in text


@pytest.mark.parametrize(
    ("side", "commit_id", "expected_tag"),
    [
        ("LEFT", "", "(base)"),
        ("RIGHT", "aabbccdd11223344", "stale:"),
    ],
)
def test_show_side_and_commit_tags(
    draft_engine: Engine, side: DiffSide, commit_id: str, expected_tag: str
) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[
                InlineComment(path="a.py", line=5, side=side, commit_id=commit_id, body="Tagged."),
            ],
        ),
    )
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert expected_tag in text


def test_show_file_level_no_line(draft_engine: Engine) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[InlineComment(path="dvc.yaml", line=0, body="Missing dep.")],
        ),
    )
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "dvc.yaml" in text
    assert ":0" not in text


def test_show_status_indicators(draft_engine: Engine) -> None:

    draft = ReviewDraft(
        summary=".",
        comments=[
            InlineComment(path="a.py", line=1, body="New comment."),
            InlineComment(path="b.py", line=2, body="Clean comment.", comment_hash=""),
        ],
    )
    draft.comments[1] = draft.comments[1].model_copy(
        update={"comment_hash": _comment_hash(draft.comments[1])}
    )
    save_draft(42, draft)
    cmd_draft(draft_engine, "")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "★" in text
    assert "✓" in text


# ── Dispatch ─────────────────────────────────────────────────────────


def test_no_pr_selected(draft_engine: Engine) -> None:
    draft_engine.state.review_target = None
    cmd_draft(draft_engine, "")
    assert "No PR selected" in "\n".join(output_texts(drain(draft_engine.events)))


@pytest.mark.parametrize("subcommand", ["bogus", "foo bar"])
def test_unknown_subcommand(draft_engine: Engine, subcommand: str) -> None:
    cmd_draft(draft_engine, subcommand)
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "Unknown subcommand" in text
    assert "Usage" in text


# ── Sync (/draft sync) ──────────────────────────────────────────────


def test_sync_no_local_no_remote_is_noop(draft_engine: Engine, mock_pr: PullRequest) -> None:
    cmd_draft(draft_engine, "sync")
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "Nothing to sync" in text
    assert mock_pr.create_review.call_count == 0  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type


def test_sync_not_authenticated(draft_engine: Engine) -> None:
    draft_engine.state.gh = None
    with pytest.raises(RbtrError, match="Not authenticated"):
        cmd_draft(draft_engine, "sync")


def test_sync_local_only_pushes(draft_engine: Engine, mock_pr: PullRequest) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="My review.",
            comments=[InlineComment(path="handler.py", line=5, body="Missing error handling.")],
        ),
    )
    cmd_draft(draft_engine, "sync")

    assert mock_pr.create_review.call_count == 1  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "Draft synced" in text


def test_sync_pulls_and_pushes(
    draft_engine: Engine,
    mock_pr: PullRequest,
    pending_review: MockReview,
    mocker: MockerFixture,
) -> None:
    """Local + remote comments both appear in the synced draft."""
    save_draft(
        42,
        ReviewDraft(
            summary="Local summary.",
            comments=[InlineComment(path="a.py", line=10, body="Local finding.")],
        ),
    )
    mock_pr.create_review.return_value = mocker.MagicMock(id=201)  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    review_comments: dict[int, list[MockPRComment]] = {
        99: [make_comment(mocker, comment_id=50, path="b.py", line=20, body="Remote finding.")],
        201: [
            make_comment(mocker, comment_id=300, path="a.py", line=10, body="Local finding."),
            make_comment(mocker, comment_id=301, path="b.py", line=20, body="Remote finding."),
        ],
    }
    mock_pr.get_single_review_comments.side_effect = lambda rid: review_comments.get(rid, [])  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(draft_engine, "sync")

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 2


def test_sync_remote_edit_accepted(
    draft_engine: Engine,
    mock_pr: PullRequest,
    pending_review: MockReview,
    mocker: MockerFixture,
) -> None:
    """Remote edit to a clean comment is accepted."""

    original = ReviewDraft(
        summary="Review.",
        comments=[InlineComment(path="a.py", line=10, body="Original.", github_id=50)],
    )
    save_draft(42, stamp_synced(original))

    mock_pr.create_review.return_value = mocker.MagicMock(id=201)  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    review_comments: dict[int, list[MockPRComment]] = {
        99: [make_comment(mocker, comment_id=50, path="a.py", line=10, body="Edited remotely.")],
        201: [make_comment(mocker, comment_id=300, path="a.py", line=10, body="Edited remotely.")],
    }
    mock_pr.get_single_review_comments.side_effect = lambda rid: review_comments.get(rid, [])  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(draft_engine, "sync")

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].body == "Edited remotely."


def test_sync_remote_summary_adopted_when_local_empty(
    draft_engine: Engine,
    mock_pr: PullRequest,
    pending_review: MockReview,
    mocker: MockerFixture,
) -> None:
    pending_review.body = "Remote summary."  # type: ignore[misc]  # mock attr via type alias
    save_draft(
        42, ReviewDraft(summary="", comments=[InlineComment(path="a.py", line=1, body="C.")])
    )
    mock_pr.create_review.return_value = mocker.MagicMock(id=201)  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    review_comments: dict[int, list[MockPRComment]] = {
        99: [],
        201: [make_comment(mocker, path="a.py", line=1, body="C.")],
    }
    mock_pr.get_single_review_comments.side_effect = lambda rid: review_comments.get(rid, [])  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(draft_engine, "sync")

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Remote summary."


def test_sync_local_summary_takes_priority(
    draft_engine: Engine,
    mock_pr: PullRequest,
    pending_review: MockReview,
    mocker: MockerFixture,
) -> None:
    pending_review.body = "Remote."  # type: ignore[misc]  # mock attr via type alias
    save_draft(
        42, ReviewDraft(summary="Local.", comments=[InlineComment(path="a.py", line=1, body="C.")])
    )
    mock_pr.create_review.return_value = mocker.MagicMock(id=201)  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    review_comments: dict[int, list[MockPRComment]] = {
        99: [],
        201: [make_comment(mocker, path="a.py", line=1, body="C.")],
    }
    mock_pr.get_single_review_comments.side_effect = lambda rid: review_comments.get(rid, [])  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(draft_engine, "sync")

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Local."


def test_sync_conflict_warns(
    draft_engine: Engine,
    mock_pr: PullRequest,
    pending_review: MockReview,
    mocker: MockerFixture,
) -> None:
    """Both local and remote edit the same comment → conflict warning."""

    original = ReviewDraft(
        summary=".",
        comments=[InlineComment(path="a.py", line=10, body="Original.", github_id=50)],
    )
    synced = stamp_synced(original)
    # Dirty the local copy
    synced = synced.model_copy(
        update={"comments": [synced.comments[0].model_copy(update={"body": "Local edit."})]}
    )
    save_draft(42, synced)

    mock_pr.create_review.return_value = mocker.MagicMock(id=201)  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    review_comments: dict[int, list[MockPRComment]] = {
        99: [make_comment(mocker, comment_id=50, path="a.py", line=10, body="Remote edit.")],
        201: [make_comment(mocker, path="a.py", line=10, body="Local edit.")],
    }
    mock_pr.get_single_review_comments.side_effect = lambda rid: review_comments.get(rid, [])  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(draft_engine, "sync")

    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "conflict" in text.lower()
    # Local edit wins
    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].body == "Local edit."


# ── Post (/draft post) ──────────────────────────────────────────────


def test_post_no_draft(draft_engine: Engine, mocker: MockerFixture) -> None:
    cmd_draft(draft_engine, "post")
    assert "No draft to post" in "\n".join(output_texts(drain(draft_engine.events)))


def test_post_empty_draft(draft_engine: Engine, mocker: MockerFixture) -> None:
    save_draft(42, ReviewDraft(summary="", comments=[]))
    cmd_draft(draft_engine, "post")
    assert "Draft is empty" in "\n".join(output_texts(drain(draft_engine.events)))


@pytest.mark.parametrize("arg", ["merge", "yolo"])
def test_post_invalid_event(draft_engine: Engine, arg: str, mocker: MockerFixture) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[InlineComment(path="a.py", line=1, body="Issue.")],
        ),
    )
    cmd_draft(draft_engine, f"post {arg}")
    assert "Unknown event type" in "\n".join(output_texts(drain(draft_engine.events)))


@pytest.mark.parametrize(
    ("arg", "expected_event"),
    [
        ("", "COMMENT"),
        ("approve", "APPROVE"),
        ("request_changes", "REQUEST_CHANGES"),
    ],
)
def test_post_event_types(
    draft_engine: Engine,
    mock_pr: PullRequest,
    arg: str,
    expected_event: str,
    mocker: MockerFixture,
) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="Looks good.",
            comments=[InlineComment(path="src/app.py", line=15, body="Nice refactor.")],
        ),
    )
    cmd_draft(draft_engine, f"post {arg}")

    mock_pr.create_review.assert_called_once()  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    call_kw = mock_pr.create_review.call_args.kwargs  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    assert call_kw["event"] == expected_event
    assert call_kw["body"] == "Looks good."
    assert len(call_kw["comments"]) == 1
    assert call_kw["comments"][0]["path"] == "src/app.py"


def test_post_sends_suggestion_in_body(
    draft_engine: Engine, mock_pr: PullRequest, mocker: MockerFixture
) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="Fix.",
            comments=[
                InlineComment(path="util.py", line=8, body="Use constant.", suggestion="MAX = 100"),
            ],
        ),
    )
    cmd_draft(draft_engine, "post")

    comments = mock_pr.create_review.call_args.kwargs["comments"]  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    assert "```suggestion" in comments[0]["body"]
    assert "MAX = 100" in comments[0]["body"]


def test_post_draft_loaded_from_disk(
    draft_engine: Engine, mock_pr: PullRequest, mocker: MockerFixture
) -> None:
    """Draft saved to disk is loaded and posted correctly."""
    save_draft(
        42,
        ReviewDraft(
            summary="Disk review.",
            comments=[InlineComment(path="x.py", line=5, body="From disk.")],
        ),
    )
    cmd_draft(draft_engine, "post")

    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "posted" in text.lower() or "review" in text.lower()
    assert mock_pr.create_review.call_args.kwargs["body"] == "Disk review."  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type


# ── Clear (/draft clear) ────────────────────────────────────────────


def test_clear_deletes_draft(draft_engine: Engine, mocker: MockerFixture) -> None:
    save_draft(
        42,
        ReviewDraft(
            summary="To be deleted.",
            comments=[InlineComment(path="a.py", line=1, body="Gone.")],
        ),
    )
    cmd_draft(draft_engine, "clear")

    assert load_draft(42) is None
    assert "deleted" in "\n".join(output_texts(drain(draft_engine.events))).lower()


# ── Sync — side/commit_id/file-level ────────────────────────────────


def test_sync_translates_stale_comment_line(
    repo_draft_engine: Engine,
    mock_pr: PullRequest,
    draft_pr_target: PRTarget,
    mocker: MockerFixture,
) -> None:
    """Comment at base line 2 gets translated to head line 3."""
    mock_pr.head.sha = draft_pr_target.head_sha  # type: ignore[misc]  # autospec mock — writing read-only property
    mock_pr.base.sha = draft_pr_target.base_commit  # type: ignore[misc]  # autospec mock — writing read-only property
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(
                    path="src/handler.py",
                    line=2,
                    body="Return statement.",
                    commit_id=draft_pr_target.base_commit,
                ),
            ],
        ),
    )
    mock_pr.create_review.return_value = mocker.MagicMock(  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        id=201, html_url="https://github.com/owner/repo/pull/42#review"
    )
    mock_pr.get_single_review_comments.side_effect = lambda rid: []  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(repo_draft_engine, "sync")

    pushed = mock_pr.create_review.call_args.kwargs["comments"]  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    assert pushed[0]["line"] == 3


def test_sync_warns_and_skips_deleted_file(
    repo_draft_engine: Engine,
    mock_pr: PullRequest,
    draft_pr_target: PRTarget,
    mocker: MockerFixture,
) -> None:
    """Comment on deleted file is warned about and skipped from push."""
    mock_pr.head.sha = draft_pr_target.head_sha  # type: ignore[misc]  # autospec mock — writing read-only property
    mock_pr.base.sha = draft_pr_target.base_commit  # type: ignore[misc]  # autospec mock — writing read-only property
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(
                    path="readme.md",
                    line=1,
                    body="Gone.",
                    commit_id=draft_pr_target.base_commit,
                ),
            ],
        ),
    )
    mock_pr.create_review.return_value = mocker.MagicMock(  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        id=201, html_url="https://github.com/owner/repo/pull/42#review"
    )
    mock_pr.get_single_review_comments.side_effect = lambda rid: []  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(repo_draft_engine, "sync")

    text = "\n".join(output_texts(drain(repo_draft_engine.events)))
    assert "deleted" in text.lower()
    pushed = mock_pr.create_review.call_args.kwargs["comments"]  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    assert len(pushed) == 0


def test_sync_sends_commit_id(
    repo_draft_engine: Engine,
    mock_pr: PullRequest,
    draft_pr_target: PRTarget,
    mocker: MockerFixture,
) -> None:
    """Push passes head_sha as commit_id to the GitHub API."""
    mock_pr.head.sha = draft_pr_target.head_sha  # type: ignore[misc]  # autospec mock — writing read-only property
    mock_pr.base.sha = draft_pr_target.base_commit  # type: ignore[misc]  # autospec mock — writing read-only property
    save_draft(42, ReviewDraft(summary=".", comments=[]))
    mock_pr.create_review.return_value = mocker.MagicMock(  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        id=201, html_url="https://github.com/owner/repo/pull/42#review"
    )
    mock_pr.get_single_review_comments.side_effect = lambda rid: []  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(repo_draft_engine, "sync")

    call_kw = mock_pr.create_review.call_args.kwargs  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    assert "commit" in call_kw


def test_sync_sends_side_per_comment(
    repo_draft_engine: Engine,
    mock_pr: PullRequest,
    draft_pr_target: PRTarget,
    mocker: MockerFixture,
) -> None:
    """LEFT and RIGHT sides are forwarded in the push payload."""
    # Align mock PR SHAs with the real repo so ref resolution works.
    mock_pr.head.sha = draft_pr_target.head_sha  # type: ignore[misc]  # autospec mock — writing read-only property
    mock_pr.base.sha = draft_pr_target.base_commit  # type: ignore[misc]  # autospec mock — writing read-only property
    head_sha = draft_pr_target.head_sha
    save_draft(
        42,
        ReviewDraft(
            summary=".",
            comments=[
                InlineComment(
                    path="src/handler.py",
                    line=2,
                    side=DiffSide.LEFT,
                    body="Old.",
                    commit_id=head_sha,
                ),
                InlineComment(
                    path="src/handler.py",
                    line=2,
                    side=DiffSide.RIGHT,
                    body="New.",
                    commit_id=head_sha,
                ),
            ],
        ),
    )
    mock_pr.create_review.return_value = mocker.MagicMock(  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        id=201, html_url="https://github.com/owner/repo/pull/42#review"
    )
    review_comments: dict[int, list[MockPRComment]] = {201: []}
    mock_pr.get_single_review_comments.side_effect = lambda rid: review_comments.get(rid, [])  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(repo_draft_engine, "sync")

    pushed = mock_pr.create_review.call_args.kwargs["comments"]  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    sides = {c["side"] for c in pushed}
    assert "LEFT" in sides
    assert "RIGHT" in sides


def test_sync_file_level_comments_pass_validation(
    draft_engine: Engine,
    mock_pr: PullRequest,
    mocker: MockerFixture,
) -> None:
    """File-level comments (line=0) are not rejected as stale."""
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(path="dvc.yaml", line=0, body="Missing dep."),
                InlineComment(path="eval/evaluate.py", line=0, body="Merge marker."),
            ],
        ),
    )
    mock_pr.create_review.return_value = mocker.MagicMock(  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        id=201, html_url="https://github.com/owner/repo/pull/42#review"
    )
    review_comments: dict[int, list[MockPRComment]] = {201: []}
    mock_pr.get_single_review_comments.side_effect = lambda rid: review_comments.get(rid, [])  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type

    cmd_draft(draft_engine, "sync")

    pushed = mock_pr.create_review.call_args.kwargs["comments"]  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    assert len(pushed) == 2
    assert "line" not in pushed[0]
    assert "side" not in pushed[0]


# ── Post — side/commit_id/file-level ────────────────────────────────


def test_post_sends_side_and_commit_id(
    draft_engine: Engine, mock_pr: PullRequest, mocker: MockerFixture
) -> None:
    """post forwards side per-comment and commit_id per-review."""
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(path="a.py", line=10, side=DiffSide.LEFT, body="Old."),
                InlineComment(path="a.py", line=20, side=DiffSide.RIGHT, body="New."),
            ],
        ),
    )
    cmd_draft(draft_engine, "post")

    call_kw = mock_pr.create_review.call_args.kwargs  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
    assert call_kw["comments"][0]["side"] == "LEFT"
    assert call_kw["comments"][1]["side"] == "RIGHT"
    # commit_id is passed as the commit kwarg
    assert "commit" in call_kw


def test_post_errors_on_deleted_lines(
    repo_draft_engine: Engine,
    mock_pr: PullRequest,
    draft_pr_target: PRTarget,
    mocker: MockerFixture,
) -> None:
    """Posting a draft with comments on deleted lines raises."""
    mock_pr.head.sha = draft_pr_target.head_sha  # type: ignore[misc]  # autospec mock — writing read-only property
    mock_pr.base.sha = draft_pr_target.base_commit  # type: ignore[misc]  # autospec mock — writing read-only property
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[
                InlineComment(
                    path="readme.md",
                    line=1,
                    body="Gone.",
                    commit_id=draft_pr_target.base_commit,
                ),
            ],
        ),
    )
    with pytest.raises(RbtrError, match="deleted lines"):
        cmd_draft(repo_draft_engine, "post")


def test_post_file_level_comments_not_rejected(draft_engine: Engine, mock_pr: PullRequest) -> None:
    """File-level comments (line=0) do not raise on post."""
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[InlineComment(path="dvc.yaml", line=0, body="Fix deps.")],
        ),
    )
    cmd_draft(draft_engine, "post")

    text = "\n".join(output_texts(drain(draft_engine.events)))
    assert "posted" in text.lower() or "review" in text.lower()
    assert len(mock_pr.create_review.call_args.kwargs["comments"]) == 1  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
