"""Handler for /review — PR and branch listing/selection."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pygit2
from github.GithubException import GithubException

from rbtr.events import ColumnDef, TableOutput
from rbtr.exceptions import RbtrError
from rbtr.git import default_branch, fetch_pr_refs, list_local_branches, resolve_commit
from rbtr.github import client
from rbtr.models import BranchTarget, PRTarget
from rbtr.styles import COLUMN_BRANCH

from .indexing import run_index
from .publish import _sync_pending_draft, _warn_access

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
            base_commit=pr.base_sha or pr.base_branch,
            head_commit=pr.head_sha or pr.head_branch,
            head_sha=pr.head_sha,
            updated_at=pr.updated_at,
        )
        engine.state.discussion_cache = None

        # Fetch the PR head commit and base branch so they're
        # available locally for indexing and tools.  Without the
        # base branch fetch, a stale remote-tracking ref causes
        # diffs and commit logs to include unrelated history.
        if engine.state.repo is not None:
            fetch_pr_refs(engine.state.repo, pr.number, pr.base_branch)
            _check_refs(
                engine,
                pr.base_sha or pr.base_branch,
                pr.head_sha or pr.head_branch,
            )

        _update_session_label(engine)
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
        base_commit=resolved_base,
        head_commit=target,
        updated_at=datetime.fromtimestamp(commit.commit_time, tz=UTC),
    )
    _update_session_label(engine)
    _print_review_target(engine)
    run_index(engine)


def _update_session_label(engine: Engine) -> None:
    """Auto-name the session after the review target, once only.

    Only fires when ``session_label`` is empty — meaning no
    previous ``/review`` or ``/session rename`` has set it.
    The empty string is the persistent sentinel: it survives
    in the DB across restarts, so ``/session resume`` followed
    by ``/review`` does the right thing without extra state.
    """
    if engine.state.session_label:
        return
    target = engine.state.review_target
    if target is None:
        return
    prefix = ""
    if engine.state.owner and engine.state.repo_name:
        prefix = f"{engine.state.owner}/{engine.state.repo_name} — "
    engine.state.session_label = f"{prefix}{target.base_branch} → {target.head_branch}"


def _check_refs(engine: Engine, base_ref: str, head_ref: str) -> None:
    """Warn if either review ref cannot be resolved locally.

    Called after ``fetch_pr_refs`` so that network-reachable refs
    are already fetched.  If a ref still can't be resolved, the
    user sees a clear warning instead of silently wrong diffs
    and commit logs.
    """
    repo = engine.state.repo
    if repo is None:
        return
    missing: list[str] = []
    for label, ref in [("base", base_ref), ("head", head_ref)]:
        try:
            resolve_commit(repo, ref)
        except KeyError:
            missing.append(f"{label} ref `{ref}`")
    if missing:
        engine._warn(
            f"Cannot resolve {' and '.join(missing)} locally. "
            f"Try `git fetch origin` and re-run /review."
        )


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
