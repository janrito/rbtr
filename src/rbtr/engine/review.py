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
from rbtr.git.objects import DiffLineRanges, diff_line_ranges, translate_line
from rbtr.github import client
from rbtr.github.draft import (
    delete_draft,
    is_tombstone,
    load_draft,
    match_comments,
    save_draft,
    stamp_synced,
)
from rbtr.models import BranchTarget, InlineComment, PRTarget, ReviewDraft, ReviewEvent
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

    local = load_draft(pr_number) or ReviewDraft()

    result = match_comments(local.comments, pending.comments)
    for w in result.warnings:
        engine._warn(w)

    # Preserve the remote review body as summary if local has none.
    summary = local.summary
    if not summary and pending.summary:
        summary = pending.summary

    merged = local.model_copy(
        update={
            "summary": summary,
            "comments": result.comments,
            "github_review_id": pending.github_review_id,
        }
    )
    # Stamp hashes so we can detect edits on next sync.
    merged = stamp_synced(merged)
    save_draft(pr_number, merged)
    n = len(merged.comments)
    engine._out(f"Draft synced from GitHub ({n} comment{'s' if n != 1 else ''}).")


def _get_diff_ranges(engine: Engine) -> DiffLineRanges:
    """Compute commentable line ranges from the review target.

    Returns an empty dict when the repo or target is unavailable
    (validation is skipped in that case).
    """
    target = engine.state.review_target
    repo = engine.state.repo
    if target is None or repo is None:
        return {}
    try:
        return diff_line_ranges(repo, target.base_branch, target.head_ref)
    except KeyError:
        return {}


def _partition_comments(
    comments: list[InlineComment],
    ranges: DiffLineRanges,
) -> tuple[list[InlineComment], list[InlineComment]]:
    """Split comments into (valid, invalid) based on diff line ranges.

    A comment is valid when its ``path`` and ``line`` appear in the
    diff.  File-level comments (``line == 0``) are always valid.
    When *ranges* is empty (no diff data), all comments are
    treated as valid — validation is best-effort.
    """
    if not ranges:
        return comments, []
    valid: list[InlineComment] = []
    invalid: list[InlineComment] = []
    for c in comments:
        if c.line == 0 or (c.path in ranges and c.line in ranges[c.path]):
            valid.append(c)
        else:
            invalid.append(c)
    return valid, invalid


def _translate_stale_comments(
    engine: Engine,
    comments: list[InlineComment],
    target_sha: str,
) -> tuple[list[InlineComment], list[InlineComment]]:
    """Translate comments whose ``commit_id`` differs from *target_sha*.

    Returns ``(translated, lost)`` where *translated* have updated
    ``line`` and ``commit_id``, and *lost* are comments whose lines
    were deleted in the newer commit.

    Comments with an empty ``commit_id`` (legacy) or already matching
    *target_sha* are returned as-is.
    """
    repo = engine.state.repo
    if repo is None or not target_sha:
        return comments, []

    translated: list[InlineComment] = []
    lost: list[InlineComment] = []
    for c in comments:
        # File-level comments have no line to translate.
        if c.line == 0 or not c.commit_id or c.commit_id == target_sha:
            translated.append(c)
            continue
        new_line = translate_line(repo, c.path, c.commit_id, target_sha, c.line)
        if new_line is None:
            lost.append(c)
        else:
            translated.append(c.model_copy(update={"line": new_line, "commit_id": target_sha}))
    return translated, lost


def post_review_draft(
    engine: Engine,
    pr_number: int,
    draft: ReviewDraft,
    event: ReviewEvent,
) -> None:
    """Post a review draft to GitHub.

    Handles the full flow: pull remote to detect unsynced comments,
    delete any existing pending review, post the submitted review,
    emit events, and clean up the local draft.  Raises
    :class:`~rbtr.exceptions.RbtrError` on failure.
    """
    ctx = engine.state.gh_ctx
    if ctx is None or not engine.state.gh_username:
        raise RbtrError("Not authenticated. Run /connect github first.")

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
        # Match remote comments against local to detect unknowns.
        result = match_comments(draft.comments, pending.comments)
        # Any remote comment that ended up imported (not matched to a
        # local comment) means there are unsynced comments.
        local_gids = {c.github_id for c in draft.comments if c.github_id is not None}
        unsynced = [
            c for c in result.comments if c.github_id is not None and c.github_id not in local_gids
        ]
        if unsynced:
            details = "\n".join(f"  {c.path}:{c.line} — {c.body[:60]}" for c in unsynced)
            raise RbtrError(
                f"Remote pending review has {len(unsynced)} comment(s) not in "
                f"your local draft. Run /draft sync first.\n{details}"
            )

    engine._clear()

    # Delete existing pending review before posting.
    review_to_delete = pending.github_review_id if pending is not None else draft.github_review_id
    if review_to_delete is not None:
        engine._out("Replacing existing pending review…")
        try:
            client.delete_pending_review(ctx, pr_number, review_to_delete)
        except GithubException as exc:
            engine._clear()
            if exc.status != 404:
                raise RbtrError(f"Failed to delete pending review: {exc.data}") from exc

    # Drop tombstones (locally deleted synced comments).
    draft = draft.model_copy(
        update={"comments": [c for c in draft.comments if not is_tombstone(c)]}
    )

    # Translate stale comments (different commit_id) to current head.
    target = engine.state.review_target
    head_sha = target.head_sha if isinstance(target, PRTarget) else ""
    if draft.comments and head_sha:
        translated, lost = _translate_stale_comments(engine, draft.comments, head_sha)
        if lost:
            details = "\n".join(f"  {c.path}:{c.line} — {c.body[:60]}" for c in lost)
            raise RbtrError(
                f"{len(lost)} comment(s) target deleted lines and cannot be "
                f"translated to the current PR head.\n{details}\n"
                f"Remove or fix them, then try again."
            )
        draft = draft.model_copy(update={"comments": translated})

    # Validate comment locations against the current diff.
    if draft.comments:
        ranges = _get_diff_ranges(engine)
        _valid, invalid = _partition_comments(draft.comments, ranges)
        if invalid:
            details = "\n".join(f"  {c.path}:{c.line} — {c.body[:60]}" for c in invalid)
            raise RbtrError(
                f"{len(invalid)} comment(s) target lines not in the PR diff "
                f"(the PR may have been updated).\n{details}\n"
                f"Remove or fix them, then try again."
            )

    # Post.
    n = len(draft.comments)
    engine._out(f"Posting review ({event.value}, {n} comment{'s' if n != 1 else ''})…")
    try:
        url = client.post_review(ctx, pr_number, draft, event, commit_id=head_sha)
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
    2. Match remote comments against local using github_id and content.
    3. Delete the old pending review.
    4. Filter tombstones (locally deleted synced comments) — they
       are excluded from the push and dropped from the draft.
    5. Validate comments against the current diff — stale comments
       (targeting lines no longer in the diff) are kept locally
       but excluded from the push.
    6. Push the merged draft as a new PENDING review.
    7. Re-fetch to learn new github_ids, update sync state.

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

    engine._clear()

    # 2. Match remote into local.
    local = load_draft(pr_number) or ReviewDraft()
    if pending is not None:
        result = match_comments(local.comments, pending.comments)
        for w in result.warnings:
            engine._warn(w)

        summary = local.summary
        if not summary and pending.summary:
            summary = pending.summary
        local = local.model_copy(update={"summary": summary, "comments": result.comments})
    else:
        engine._out(f"No pending review found for '{engine.state.gh_username}'.")

    if not local.summary and not local.comments:
        engine._out("Nothing to sync — no local or remote draft.")
        return

    # 3. Delete existing pending review before pushing.
    review_to_delete = pending.github_review_id if pending is not None else local.github_review_id
    if review_to_delete is not None:
        try:
            client.delete_pending_review(ctx, pr_number, review_to_delete)
        except GithubException as exc:
            if exc.status != 404:
                raise RbtrError(f"Failed to delete pending review: {exc.data}") from exc
            # 404 = already gone (crash recovery). Continue.

    # 4. Translate stale comments, then validate against the diff.
    target = engine.state.review_target
    head_sha = target.head_sha if isinstance(target, PRTarget) else ""
    if local.comments and head_sha:
        translated, lost = _translate_stale_comments(engine, local.comments, head_sha)
        for c in lost:
            engine._warn(f"Skipped comment ({c.path}:{c.line}) — line deleted in current head.")
        local = local.model_copy(update={"comments": translated})

    # Filter tombstones (locally deleted synced comments).  They are
    # excluded from the push so the remote copy is not recreated.
    # Dropped entirely after the push — not saved back to the draft.
    tombstones = [c for c in local.comments if is_tombstone(c)]
    live_comments = [c for c in local.comments if not is_tombstone(c)]
    if tombstones:
        n_t = len(tombstones)
        engine._out(f"Deleting {n_t} comment{'s' if n_t != 1 else ''} from remote review.")

    ranges = _get_diff_ranges(engine)
    pushable, stale = _partition_comments(live_comments, ranges)
    for c in stale:
        engine._warn(f"Skipped stale comment ({c.path}:{c.line}) — line not in current diff.")

    push_draft = local.model_copy(update={"comments": pushable})

    # 5. Push merged draft as new PENDING.
    n = len(pushable)
    engine._out(f"Pushing draft ({n} comment{'s' if n != 1 else ''})…")
    try:
        new_review_id = client.push_pending_review(ctx, pr_number, push_draft, commit_id=head_sha)
    except GithubException as exc:
        engine._clear()
        raise RbtrError(f"Failed to push draft: {exc.data}") from exc

    # 6. Re-fetch to learn new github_ids.
    try:
        pushed = client.get_pending_review(ctx, pr_number, engine.state.gh_username)
    except GithubException:
        pushed = None

    if pushed is not None:
        # Match by content to assign github_ids to pushed comments.
        # Strip comment_hash and github_id so we get pure tier-2 matching
        # (the old IDs are gone after delete-and-recreate).
        clean = [c.model_copy(update={"comment_hash": "", "github_id": None}) for c in pushable]
        rematched = match_comments(clean, pushed.comments)
        synced_comments = rematched.comments
    else:
        synced_comments = pushable

    # Merge synced (pushed) and stale (skipped) comments back into
    # the local draft.  Stale comments keep their previous state and
    # are appended after pushed comments so they remain visible.
    all_comments = synced_comments + stale
    local = local.model_copy(update={"comments": all_comments, "github_review_id": new_review_id})
    local = stamp_synced(local)
    save_draft(pr_number, local)

    engine._clear()
    parts: list[str] = [f"Draft synced ({n} comment{'s' if n != 1 else ''} pushed)"]
    if stale:
        parts.append(f"{len(stale)} stale comment{'s' if len(stale) != 1 else ''} kept locally")
    engine._out(". ".join(parts) + ".")


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

    if pending is not None and pending.github_review_id is not None:
        try:
            client.delete_pending_review(ctx, pr_number, pending.github_review_id)
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
