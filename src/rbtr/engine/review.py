"""Handler for /review — PR and branch listing/selection, draft posting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pygit2
from github import UnknownObjectException
from github.GithubException import GithubException

from rbtr.events import ColumnDef, LinkOutput, ReviewPosted, TableOutput
from rbtr.exceptions import RbtrError
from rbtr.git import default_branch, fetch_pr_head, list_local_branches
from rbtr.github import client
from rbtr.github.draft import (
    delete_draft,
    get_unsynced_comments,
    load_draft,
    merge_remote,
    save_draft,
)
from rbtr.models import BranchTarget, PRTarget, ReviewDraft, ReviewEvent
from rbtr.styles import COLUMN_BRANCH, LINK_STYLE

from .indexing import run_index

if TYPE_CHECKING:
    from .core import Engine


def cmd_review(engine: Engine, identifier: str) -> None:
    """List open PRs/branches, or select a review target.

    Syntax::

        /review                     — list open PRs and branches
        /review <number>            — select a GitHub PR
        /review <branch>            — review branch vs default base
        /review <base> <target>     — review target vs explicit base
    """
    if not identifier:
        _list(engine)
        return

    parts = identifier.split()
    if len(parts) == 1:
        try:
            pr_number = int(parts[0])
            _review_pr(engine, pr_number)
            return
        except ValueError:
            pass
        _review_branch(engine, base=None, target=parts[0])
    elif len(parts) == 2:
        _review_branch(engine, base=parts[0], target=parts[1])
    else:
        engine._warn("Usage: /review [base] <target> or /review <pr_number>")


def _list(engine: Engine) -> None:
    ctx = engine.state.gh_ctx
    if ctx is not None:
        try:
            engine._out("Fetching from GitHub…")
            prs = client.list_open_prs(ctx)
            engine._check_cancel()
            pr_branches = {pr.head_branch for pr in prs}
            branches = client.list_unmerged_branches(ctx, pr_branches)
            engine._clear()

            if not prs and not branches:
                engine._out(f"No open PRs or unmerged branches in {ctx.full_name}.")
                return

            if prs:
                engine._emit(
                    TableOutput(
                        title="Open Pull Requests",
                        columns=[
                            ColumnDef(header="PR", width=8),
                            ColumnDef(header="Title"),
                            ColumnDef(header="Author", width=16),
                            ColumnDef(header="Branch", style=COLUMN_BRANCH),
                        ],
                        rows=[[f"#{pr.number}", pr.title, pr.author, pr.head_branch] for pr in prs],
                    )
                )

            if prs and branches:
                engine._flush()

            if branches:
                engine._emit(
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

            # Cache for Tab completion.
            targets: list[tuple[str, str]] = []
            for pr in prs:
                targets.append((f"#{pr.number} {pr.title}", str(pr.number)))
            for b in branches:
                targets.append((b.name, b.name))
            engine.state.cached_review_targets = targets

            engine._out("Use /review <pr_number> or /review <branch_name> to select.")
            return
        except GithubException as e:
            engine._clear()
            if e.status in (403, 404):
                _warn_access(engine, e)
            else:
                engine._warn(f"GitHub error ({e.status}): {e.data}")
                engine._out("Falling back to local branches.")

    if engine.state.repo is None:
        return
    branches_local = list_local_branches(engine.state.repo)
    if not branches_local:
        engine._out("No local branches found.")
        return

    engine._emit(
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
    # Cache for Tab completion.
    engine.state.cached_review_targets = [(b.name, b.name) for b in branches_local]

    engine._out("Use /review <branch_name> to select.")


def _review_pr(engine: Engine, pr_number: int) -> None:
    ctx = engine.state.gh_ctx
    if ctx is None:
        engine._warn("Not authenticated. Run /connect github first.")
        return
    try:
        engine._out(f"Fetching PR #{pr_number}…")
        pr = client.validate_pr_number(ctx, pr_number)
        engine._clear()
        engine.state.review_target = PRTarget(
            number=pr.number,
            title=pr.title,
            author=pr.author,
            body=pr.body,
            base_branch=pr.base_branch,
            head_branch=pr.head_branch,
            head_sha=pr.head_sha,
            updated_at=pr.updated_at,
        )
        engine.state.discussion_cache = None

        # Fetch the PR head commit so it's available locally for
        # indexing and tools (works for forks and unfetched branches).
        if engine.state.repo is not None:
            fetch_pr_head(engine.state.repo, pr.number)

        _print_review_target(engine)
        _sync_pending_draft(engine, pr.number)
        run_index(engine)
    except GithubException as e:
        engine._clear()
        engine._warn(f"Could not fetch PR #{pr_number}: {e.data}")
    except RbtrError as e:
        engine._clear()
        engine._warn(str(e))


def _review_branch(engine: Engine, *, base: str | None, target: str) -> None:
    if engine.state.repo is None:
        return

    repo = engine.state.repo

    # Validate target branch exists locally.
    if target not in repo.branches.local:
        engine._warn(f"Branch '{target}' not found locally.")
        return

    # Resolve base: explicit arg, or fall back to repo default.
    resolved_base = base if base is not None else default_branch(repo)
    if resolved_base not in repo.branches.local:
        engine._warn(f"Base branch '{resolved_base}' not found locally.")
        return

    branch = repo.branches.local[target]
    commit = branch.peel(pygit2.Commit)
    engine.state.review_target = BranchTarget(
        base_branch=resolved_base,
        head_branch=target,
        updated_at=datetime.fromtimestamp(commit.commit_time, tz=UTC),
    )
    _print_review_target(engine)
    run_index(engine)


def _print_review_target(engine: Engine) -> None:
    target = engine.state.review_target
    if target is None:
        engine._out("No review target selected. Use /review to select one.")
        return
    match target:
        case PRTarget(number=number, title=title, base_branch=base, head_branch=head):
            engine._out(f"Reviewing PR #{number}: {title} ({base} → {head})")
        case BranchTarget(base_branch=base, head_branch=head):
            engine._out(f"Reviewing branch: {base} → {head}")


def _sync_pending_draft(engine: Engine, pr_number: int) -> None:
    """Download the user's pending review from GitHub and merge into local draft.

    Pull-only — called automatically on ``/review <n>`` to seed the
    local draft.  For bidirectional sync, use ``sync_review_draft``.
    """
    ctx = engine.state.gh_ctx
    if ctx is None or not engine.state.gh_username:
        return

    try:
        pending = client.get_pending_review(ctx, pr_number, engine.state.gh_username)
    except UnknownObjectException:
        # No pending review on this PR — not an error.
        return
    except GithubException as exc:
        engine._warn(f"Could not sync pending review: {exc.data}")
        return

    if pending is None:
        # No remote pending review — check for local draft only.
        local = load_draft(pr_number)
        if local is not None:
            n = len(local.comments)
            engine._out(f"Local draft loaded ({n} comment{'s' if n != 1 else ''}).")
        return

    local = load_draft(pr_number)
    merged = merge_remote(local, pending.comments)

    # Preserve the remote review body as summary if local has none.
    if not merged.summary and pending.body:
        merged = merged.model_copy(update={"summary": pending.body})

    save_draft(pr_number, merged)
    n = len(merged.comments)
    engine._out(f"Draft synced from GitHub ({n} comment{'s' if n != 1 else ''}).")


def post_review_draft(
    engine: Engine,
    pr_number: int,
    draft: ReviewDraft,
    event: ReviewEvent,
) -> None:
    """Post a review draft to GitHub.

    Handles the full flow: guard against unsynced remote comments,
    delete any existing pending review, post the new review, emit
    events, and clean up the local draft.  Raises
    :class:`~rbtr.exceptions.RbtrError` on failure.
    """
    ctx = engine.state.gh_ctx
    if ctx is None or not engine.state.gh_username:
        raise RbtrError("Not authenticated. Run /connect github first.")

    draft = draft.model_copy(update={"event": event})

    # Guard: check for unsynced remote comments.
    engine._out("Checking for unsynced remote comments…")
    try:
        pending = client.get_pending_review(
            ctx,
            pr_number,
            engine.state.gh_username,
        )
    except GithubException as exc:
        engine._clear()
        raise RbtrError(f"GitHub error checking pending review: {exc.data}") from exc

    if pending is not None:
        unsynced = get_unsynced_comments(draft, pending.comments)
        if unsynced:
            details = "\n".join(f"  {c.path}:{c.line} — {c.body[:60]}" for c in unsynced)
            raise RbtrError(
                f"Remote pending review has {len(unsynced)} comment(s) not in "
                f"your local draft. Run /draft sync first.\n{details}"
            )

    engine._clear()

    # Delete existing pending review before posting.
    if pending is not None:
        engine._out("Replacing existing pending review…")
        try:
            client.delete_pending_review(ctx, pr_number, pending.review_id)
        except GithubException as exc:
            engine._clear()
            raise RbtrError(f"Failed to delete pending review: {exc.data}") from exc

    # Post.
    n = len(draft.comments)
    engine._out(f"Posting review ({event.value}, {n} comment{'s' if n != 1 else ''})…")
    try:
        url = client.post_review(ctx, pr_number, draft)
    except GithubException as exc:
        engine._clear()
        raise RbtrError(f"Failed to post review: {exc.data}") from exc

    engine._clear()
    engine._emit(ReviewPosted(url=url))
    engine._emit(
        LinkOutput(markup=(f"Review posted: [link={url}][{LINK_STYLE}]{url}[/{LINK_STYLE}][/link]"))
    )

    # Clean up local draft.
    delete_draft(pr_number)
    engine._out("Local draft cleared.")


def sync_review_draft(engine: Engine, pr_number: int) -> None:
    """Bidirectional sync: pull remote pending comments, push local back.

    1. Fetch the user's PENDING review from GitHub.
    2. Merge any new remote comments into the local draft.
    3. Delete the old pending review.
    4. Push the merged local draft back as a new PENDING review.

    If there is no local draft and no remote pending review,
    this is a no-op.  Raises :class:`~rbtr.exceptions.RbtrError`
    on failure so the command is marked as failed.
    """
    ctx = engine.state.gh_ctx
    if ctx is None or not engine.state.gh_username:
        raise RbtrError("Not authenticated. Run /connect github first.")

    # 1. Pull: fetch remote pending review.
    engine._out("Pulling remote pending review…")
    try:
        pending = client.get_pending_review(
            ctx,
            pr_number,
            engine.state.gh_username,
        )
    except GithubException as exc:
        engine._clear()
        raise RbtrError(f"GitHub error fetching pending review: {exc.data}") from exc

    # 2. Merge remote into local.
    local = load_draft(pr_number)
    if pending is not None:
        merged = merge_remote(local, pending.comments)
        if not merged.summary and pending.body:
            merged = merged.model_copy(update={"summary": pending.body})
        save_draft(pr_number, merged)
        local = merged

    if local is None or (not local.summary and not local.comments):
        engine._clear()
        engine._out("Nothing to sync — no local or remote draft.")
        return

    engine._clear()

    # 3. Delete existing pending review before pushing.
    if pending is not None:
        try:
            client.delete_pending_review(ctx, pr_number, pending.review_id)
        except GithubException as exc:
            raise RbtrError(f"Failed to delete pending review: {exc.data}") from exc

    # 4. Push merged draft as new PENDING.
    n = len(local.comments)
    engine._out(f"Pushing draft ({n} comment{'s' if n != 1 else ''})…")
    try:
        client.push_pending_review(ctx, pr_number, local)
    except GithubException as exc:
        engine._clear()
        raise RbtrError(f"Failed to push draft: {exc.data}") from exc

    engine._clear()
    engine._out(f"Draft synced ({n} comment{'s' if n != 1 else ''}).")


def clear_review_draft(engine: Engine, pr_number: int) -> None:
    """Delete the local draft file and any remote pending review."""
    if delete_draft(pr_number):
        engine._out("Local draft deleted.")
    else:
        engine._out("No local draft to delete.")

    ctx = engine.state.gh_ctx
    if ctx is None or not engine.state.gh_username:
        return

    try:
        pending = client.get_pending_review(
            ctx,
            pr_number,
            engine.state.gh_username,
        )
    except GithubException:
        return

    if pending is not None:
        try:
            client.delete_pending_review(ctx, pr_number, pending.review_id)
            engine._out("Remote pending review deleted.")
        except GithubException as exc:
            engine._warn(f"Failed to delete remote pending review: {exc.data}")


def _warn_access(engine: Engine, exc: GithubException) -> None:
    ctx = engine.state.gh_ctx
    name = ctx.full_name if ctx else f"{engine.state.owner}/{engine.state.repo_name}"
    engine._warn(f"Cannot access {name} via GitHub API ({exc.status}).")
    message = exc.data.get("message", "") if isinstance(exc.data, dict) else ""
    if message:
        engine._out(message)
    engine._out("Falling back to local branches.")
