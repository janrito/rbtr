"""Review draft publishing — post, sync, and clear review drafts on GitHub.

Orchestration layer between local draft persistence
(:mod:`rbtr.github.draft`) and the GitHub API
(:mod:`rbtr.github.client`).  Called by :mod:`rbtr.engine.draft_cmd`
and :mod:`rbtr.engine.review_cmd`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from github import UnknownObjectException
from github.GithubException import GithubException

from rbtr.events import LinkOutput, ReviewPosted
from rbtr.exceptions import RbtrError
from rbtr.git.objects import (
    DiffLineRanges,
    SideDiffRanges,
    diff_line_ranges,
    nearest_commentable_line,
    translate_line,
)
from rbtr.git.repo import fetch_pr_refs
from rbtr.github import client
from rbtr.github.draft import (
    delete_draft,
    is_tombstone,
    load_draft,
    match_comments,
    partition_comments,
    save_draft,
    stamp_synced,
)
from rbtr.models import DiffSide, InlineComment, PRTarget, ReviewDraft, ReviewEvent

from .setup import ensure_gh_username

if TYPE_CHECKING:
    from .core import Engine


def _sync_pending_draft(engine: Engine, pr_number: int) -> None:
    """Download the user's pending review from GitHub and merge into local draft.

    Pull-only — called automatically on ``/review <n>`` to seed the
    local draft.  For bidirectional sync, use ``sync_review_draft``.
    """
    ctx = engine.state.gh_ctx
    gh_user = ensure_gh_username(engine)
    if ctx is None or not gh_user:
        return

    try:
        pending = client.get_pending_review(ctx, pr_number, gh_user)
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


def _refresh_pr_refs(engine: Engine, pr_number: int) -> str:
    """Re-fetch live PR head and base SHAs, update local refs if changed.

    Compares the live head against the stored ``PRTarget.head_sha``
    and the live base against ``PRTarget.base_commit``.  When either
    differs, fetches new git objects and updates the session's review
    target so translation, validation, and posting all use the
    current state.

    Both refs matter: the head determines line translation, and the
    base determines the merge-base for diff computation.  If only the
    head is refreshed but the base branch has advanced, the local
    three-dot diff diverges from GitHub's.

    Returns the (possibly updated) head SHA.
    """
    ctx = engine.state.gh_ctx
    target = engine.state.review_target
    if ctx is None or not isinstance(target, PRTarget):
        return target.head_sha if isinstance(target, PRTarget) else ""

    try:
        live_head, live_base = client.get_pr_refs(ctx, pr_number)
    except GithubException:
        return target.head_sha  # best effort — use cached

    head_changed = live_head and live_head != target.head_sha
    base_changed = live_base and live_base != target.base_commit

    if not head_changed and not base_changed:
        return target.head_sha

    parts: list[str] = []
    if head_changed:
        parts.append(f"head {target.head_sha[:12]} → {live_head[:12]}")
    if base_changed:
        parts.append(f"base {target.base_commit[:12]} → {live_base[:12]}")
    engine._out(f"PR refs updated ({', '.join(parts)}). Refreshing local objects…")

    # Fetch new git objects so translate_line and diff_line_ranges work.
    repo = engine.state.repo
    if repo is not None:
        fetch_pr_refs(repo, pr_number, target.base_branch)

    # Update session target.
    updates: dict[str, str] = {}
    if head_changed:
        updates["head_sha"] = live_head
        updates["head_commit"] = live_head
    if base_changed:
        updates["base_commit"] = live_base
    engine.state.review_target = target.model_copy(update=updates)
    engine.state.diff_range_cache = None
    return updates.get("head_sha", target.head_sha)


def _local_diff_ranges(engine: Engine) -> SideDiffRanges | None:
    """Compute commentable line ranges for both diff sides from local git.

    Returns ``None`` when the repo or target refs are unavailable.
    Used for line translation and as a fallback when the GitHub API
    is unreachable.  For comment validation before push, prefer
    :func:`_github_diff_ranges` — the local diff can diverge
    from GitHub's (different merge-base resolution, diff algorithm,
    or incomplete fetch).
    """
    target = engine.state.review_target
    repo = engine.state.repo
    if not isinstance(target, PRTarget) or repo is None:
        return None
    try:
        return diff_line_ranges(repo, target.base_commit, target.head_commit)
    except KeyError:
        return None


def _github_diff_ranges(engine: Engine, pr_number: int) -> SideDiffRanges | None:
    """Fetch commentable line ranges from GitHub's own patch data.

    This is the authoritative source for which lines the review
    API will accept.  Returns ``None`` when the GitHub context is
    unavailable or the API call fails.
    """
    ctx = engine.state.gh_ctx
    if ctx is None:
        return None
    try:
        return client.get_pr_diff_ranges(ctx, pr_number)
    except Exception:
        return None


def _stale_comment_warning(
    comment: InlineComment,
    right_ranges: DiffLineRanges,
    left_ranges: DiffLineRanges,
) -> str:
    """Build a warning for a comment that failed diff-range validation."""
    ranges = left_ranges if comment.side is DiffSide.LEFT else right_ranges
    nearest = nearest_commentable_line(ranges, comment.path, comment.line)
    hint = f" Nearest commentable line: {nearest}." if nearest is not None else ""
    return f"Skipped comment ({comment.path}:{comment.line}) — line not in a diff hunk.{hint}"


def _github_error_strings(exc: GithubException) -> list[str]:
    """Extract string entries from GitHub error payloads."""
    data = exc.data
    if not isinstance(data, dict):
        return []
    raw_errors = data.get("errors")
    if not isinstance(raw_errors, list):
        return []
    return [entry for entry in raw_errors if isinstance(entry, str)]


def _is_line_resolution_error(exc: GithubException) -> bool:
    """Return ``True`` for GitHub 422 "line could not be resolved" errors."""
    if exc.status != 422:
        return False
    return any("line could not be resolved" in err.lower() for err in _github_error_strings(exc))


def _check_comment_ranges(
    comments: list[InlineComment],
    right_ranges: DiffLineRanges,
    left_ranges: DiffLineRanges,
    label: str,
    *,
    show_commit: bool = False,
) -> list[str]:
    """Check each comment against ranges, return one diagnostic line per comment."""
    out: list[str] = []
    for comment in comments:
        side_ranges = left_ranges if comment.side is DiffSide.LEFT else right_ranges
        nearest = nearest_commentable_line(side_ranges, comment.path, comment.line)
        if nearest is None:
            reason = f"file is not in {label} for this side"
        elif comment.line in side_ranges[comment.path]:
            reason = f"line is present in {label}"
        else:
            reason = f"line not in {label} (nearest: {nearest})"
        entry = f"  - {comment.path}:{comment.line} ({comment.side}) — {reason}"
        if show_commit:
            commit = comment.commit_id[:12] if comment.commit_id else "<empty>"
            entry += f"; comment commit_id: {commit}"
        out.append(entry)
    return out


def _line_resolution_diagnostics(
    engine: Engine,
    pr_number: int,
    comments: list[InlineComment],
    commit_id: str,
) -> str:
    """Build per-comment diagnostics for unresolved-line GitHub errors."""
    inline_comments = [c for c in comments if c.line > 0]
    if not inline_comments:
        return ""

    commit_label = commit_id[:12] if commit_id else "<empty>"
    lines: list[str] = ["Line-resolution diagnostics:", f"  Review commit_id: {commit_label}"]

    # Local git ranges.
    ranges = _local_diff_ranges(engine)
    if ranges is None:
        lines.append("  Local check: could not compute diff ranges (refs missing or stale).")
    else:
        lines.append("  Local check:")
        lines.extend(
            _check_comment_ranges(inline_comments, *ranges, "local diff ranges", show_commit=True)
        )

    # GitHub patch ranges.
    gh_ctx = engine.state.gh_ctx
    if gh_ctx is None:
        return "\n".join(lines)

    try:
        gh_right, gh_left = client.get_pr_diff_ranges(gh_ctx, pr_number)
    except Exception as exc:  # diagnostics best effort
        lines.append(f"  GitHub check: could not fetch PR patch ranges ({type(exc).__name__}).")
        return "\n".join(lines)

    lines.append("  GitHub check:")
    lines.extend(
        _check_comment_ranges(inline_comments, gh_right, gh_left, "GitHub PR patch ranges")
    )

    return "\n".join(lines)


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

    Handles the full flow: refresh PR refs, pull remote to detect
    unsynced comments, delete any existing pending review, translate
    and validate comments, post the submitted review, emit events,
    and clean up the local draft.  Raises
    :class:`~rbtr.exceptions.RbtrError` on failure.
    """
    ctx = engine.state.gh_ctx
    gh_user = ensure_gh_username(engine)
    if ctx is None or not gh_user:
        raise RbtrError("Not authenticated. Run /connect github first.")

    # Refresh PR head first — everything downstream must use current refs.
    head_sha = _refresh_pr_refs(engine, pr_number)

    # Guard: check for unsynced remote comments.
    engine._out("Checking for unsynced remote comments…")
    try:
        pending = client.get_pending_review(
            ctx,
            pr_number,
            gh_user,
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

    # Validate comment locations against GitHub's diff (authoritative).
    # Falls back to local diff if the API is unreachable.
    has_line_comments = any(c.line > 0 for c in draft.comments)
    if has_line_comments:
        ranges = _github_diff_ranges(engine, pr_number) or _local_diff_ranges(engine)
        if ranges is not None:
            right_ranges, left_ranges = ranges
            _valid, invalid = partition_comments(draft.comments, right_ranges, left_ranges)
            if invalid:
                details = "\n".join(
                    f"  {_stale_comment_warning(c, right_ranges, left_ranges)} — {c.body[:60]}"
                    for c in invalid
                )
                raise RbtrError(
                    f"{len(invalid)} comment(s) target lines not in a diff hunk.\n"
                    f"{details}\n"
                    f"Move them to commentable lines or remove them."
                )
        else:
            engine._warn("Cannot validate comment locations — skipping preflight.")

    # Post.
    n = len(draft.comments)
    engine._out(f"Posting review ({event.value}, {n} comment{'s' if n != 1 else ''})…")
    try:
        url = client.post_review(ctx, pr_number, draft, event, commit_id=head_sha)
    except GithubException as exc:
        engine._clear()
        message = f"Failed to post review: {exc.data}"
        if _is_line_resolution_error(exc):
            diagnostics = _line_resolution_diagnostics(engine, pr_number, draft.comments, head_sha)
            if diagnostics:
                message = f"{message}\n{diagnostics}"
        raise RbtrError(message) from exc

    engine._clear()
    engine._emit(ReviewPosted(url=url))
    engine._emit(LinkOutput(url=url, label="Review posted"))
    engine._context(
        f"[/draft post → {event.value}]",
        f"Posted review to GitHub as {event.value} ({n} comments).",
    )

    # Clean up local draft.
    delete_draft(pr_number)
    engine._out("Local draft cleared.")


def sync_review_draft(engine: Engine, pr_number: int) -> None:
    """Bidirectional sync: pull remote pending comments, push local back.

    1. Refresh PR head — everything downstream uses current refs.
    2. Fetch the user's PENDING review from GitHub.
    3. Match remote comments against local using github_id and content.
    4. Delete the old pending review.
    5. Translate stale comments to current head, filter tombstones.
    6. Validate comments against the current diff — stale comments
       (targeting lines no longer in the diff) are kept locally
       but excluded from the push.
    7. Push the merged draft as a new PENDING review.
    8. Re-fetch to learn new github_ids, update sync state.

    If there is no local draft and no remote pending review,
    this is a no-op.  Raises :class:`~rbtr.exceptions.RbtrError`
    on failure so the command is marked as failed.
    """
    ctx = engine.state.gh_ctx
    gh_user = ensure_gh_username(engine)
    if ctx is None or not gh_user:
        raise RbtrError("Not authenticated. Run /connect github first.")

    # 1. Refresh PR head first — everything downstream must use current refs.
    head_sha = _refresh_pr_refs(engine, pr_number)

    # 2. Pull: fetch remote pending review.
    engine._out("Pulling remote pending review…")
    try:
        pending = client.get_pending_review(
            ctx,
            pr_number,
            gh_user,
        )
    except GithubException as exc:
        engine._clear()
        raise RbtrError(f"GitHub error fetching pending review: {exc.data}") from exc

    engine._clear()

    # 3. Match remote into local.
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
        engine._out(f"No pending review found for '{gh_user}'.")

    if not local.summary and not local.comments:
        engine._out("Nothing to sync — no local or remote draft.")
        return

    # 4. Delete existing pending review before pushing.
    review_to_delete = pending.github_review_id if pending is not None else local.github_review_id
    if review_to_delete is not None:
        try:
            client.delete_pending_review(ctx, pr_number, review_to_delete)
        except GithubException as exc:
            if exc.status != 404:
                raise RbtrError(f"Failed to delete pending review: {exc.data}") from exc
            # 404 = already gone (crash recovery). Continue.

    # 5. Translate stale comments to current head.
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

    # 6. Validate comments against GitHub's diff (authoritative).
    # Falls back to local diff if the API is unreachable.
    right_ranges: DiffLineRanges = {}
    left_ranges: DiffLineRanges = {}
    has_line_comments = any(c.line > 0 for c in live_comments)
    if has_line_comments:
        ranges = _github_diff_ranges(engine, pr_number) or _local_diff_ranges(engine)
        if ranges is not None:
            right_ranges, left_ranges = ranges
            pushable, stale = partition_comments(live_comments, right_ranges, left_ranges)
        else:
            engine._warn("Cannot validate comment locations — skipping preflight.")
            pushable, stale = live_comments, []
    else:
        pushable, stale = live_comments, []

    push_draft = local.model_copy(update={"comments": pushable})

    # 7. Push merged draft as new PENDING.
    n = len(pushable)
    engine._out(f"Pushing draft ({n} comment{'s' if n != 1 else ''})…")
    try:
        new_review_id = client.push_pending_review(ctx, pr_number, push_draft, commit_id=head_sha)
    except GithubException as exc:
        engine._clear()
        message = f"Failed to push draft: {exc.data}"
        if _is_line_resolution_error(exc):
            diagnostics = _line_resolution_diagnostics(
                engine,
                pr_number,
                push_draft.comments,
                head_sha,
            )
            if diagnostics:
                message = f"{message}\n{diagnostics}"
        raise RbtrError(message) from exc

    # 8. Fetch comments from the just-created review to learn github_ids.
    try:
        pushed_comments = client.get_review_comments(ctx, pr_number, new_review_id)
    except GithubException:
        pushed_comments = None

    if pushed_comments is not None:
        # Match by content to assign github_ids to pushed comments.
        # Strip comment_hash and github_id so we get pure tier-2 matching
        # (the old IDs are gone after delete-and-recreate).
        clean = [c.model_copy(update={"comment_hash": "", "github_id": None}) for c in pushable]
        rematched = match_comments(clean, pushed_comments)
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
    for c in stale:
        engine._warn(_stale_comment_warning(c, right_ranges, left_ranges))
    engine._context("[/draft sync → synced]", f"Synced draft with GitHub: {n} comments pushed.")


def clear_review_draft(engine: Engine, pr_number: int) -> None:
    """Delete the local draft file and any remote pending review."""
    if delete_draft(pr_number):
        engine._out("Local draft deleted.")
    else:
        engine._out("No local draft to delete.")

    ctx = engine.state.gh_ctx
    gh_user = ensure_gh_username(engine)
    if ctx is None or not gh_user:
        return

    try:
        pending = client.get_pending_review(
            ctx,
            pr_number,
            gh_user,
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
    """Warn about GitHub access issues and suggest fallback."""
    ctx = engine.state.gh_ctx
    name = ctx.full_name if ctx else f"{engine.state.owner}/{engine.state.repo_name}"
    engine._warn(f"Cannot access {name} via GitHub API ({exc.status}).")
    message = exc.data.get("message", "") if isinstance(exc.data, dict) else ""
    if message:
        engine._out(message)
    engine._out("Falling back to local branches.")
