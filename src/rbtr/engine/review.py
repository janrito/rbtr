"""Handler for /review — PR and branch listing/selection."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pygit2
from github.GithubException import GithubException

from rbtr.events import ColumnDef, TableOutput
from rbtr.exceptions import RbtrError
from rbtr.github import client
from rbtr.models import BranchTarget, PRTarget
from rbtr.repo import default_branch, list_local_branches
from rbtr.styles import COLUMN_BRANCH

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
    if engine.session.gh is not None:
        try:
            engine._out("Fetching from GitHub…")
            prs = client.list_open_prs(
                engine.session.gh, engine.session.owner, engine.session.repo_name
            )
            engine._check_cancel()
            pr_branches = {pr.head_branch for pr in prs}
            branches = client.list_unmerged_branches(
                engine.session.gh,
                engine.session.owner,
                engine.session.repo_name,
                pr_branches,
            )
            engine._clear()

            if not prs and not branches:
                engine._out(
                    f"No open PRs or unmerged branches in "
                    f"{engine.session.owner}/{engine.session.repo_name}."
                )
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
            engine.session.cached_review_targets = targets

            engine._out("Use /review <pr_number> or /review <branch_name> to select.")
            return
        except GithubException as e:
            engine._clear()
            if e.status in (403, 404):
                _warn_access(engine, e)
            else:
                engine._warn(f"GitHub error ({e.status}): {e.data}")
                engine._out("Falling back to local branches.")

    if engine.session.repo is None:
        return
    branches_local = list_local_branches(engine.session.repo)
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
    engine.session.cached_review_targets = [(b.name, b.name) for b in branches_local]

    engine._out("Use /review <branch_name> to select.")


def _review_pr(engine: Engine, pr_number: int) -> None:

    if engine.session.gh is None:
        engine._warn("Not authenticated. Run /connect github first.")
        return
    try:
        engine._out(f"Fetching PR #{pr_number}…")
        pr = client.validate_pr_number(
            engine.session.gh, engine.session.owner, engine.session.repo_name, pr_number
        )
        engine._clear()
        engine.session.review_target = PRTarget(
            number=pr.number,
            title=pr.title,
            author=pr.author,
            body=pr.body,
            base_branch=pr.base_branch,
            head_branch=pr.head_branch,
            updated_at=pr.updated_at,
        )
        _print_review_target(engine)
        run_index(engine)
    except GithubException as e:
        engine._clear()
        engine._warn(f"Could not fetch PR #{pr_number}: {e.data}")
    except RbtrError as e:
        engine._clear()
        engine._warn(str(e))


def _review_branch(engine: Engine, *, base: str | None, target: str) -> None:
    if engine.session.repo is None:
        return

    repo = engine.session.repo

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
    engine.session.review_target = BranchTarget(
        base_branch=resolved_base,
        head_branch=target,
        updated_at=datetime.fromtimestamp(commit.commit_time, tz=UTC),
    )
    _print_review_target(engine)
    run_index(engine)


def _print_review_target(engine: Engine) -> None:
    target = engine.session.review_target
    if target is None:
        engine._out("No review target selected. Use /review to select one.")
        return
    match target:
        case PRTarget(number=number, title=title, base_branch=base, head_branch=head):
            engine._out(f"Reviewing PR #{number}: {title} ({base} → {head})")
        case BranchTarget(base_branch=base, head_branch=head):
            engine._out(f"Reviewing branch: {base} → {head}")


def _warn_access(engine: Engine, exc: GithubException) -> None:
    engine._warn(
        f"Cannot access {engine.session.owner}/{engine.session.repo_name} "
        f"via GitHub API ({exc.status})."
    )
    message = exc.data.get("message", "") if isinstance(exc.data, dict) else ""
    if message:
        engine._out(message)
    engine._out("Falling back to local branches.")
