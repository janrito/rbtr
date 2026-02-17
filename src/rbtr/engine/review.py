"""Handler for /review — PR and branch listing/selection."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pygit2
from github.GithubException import GithubException

from rbtr import RbtrError
from rbtr.events import ColumnDef, TableOutput
from rbtr.github import client
from rbtr.models import BranchTarget, PRTarget
from rbtr.repo import list_local_branches
from rbtr.styles import COLUMN_BRANCH

if TYPE_CHECKING:
    from .core import Engine


def cmd_review(engine: Engine, identifier: str) -> None:
    """List open PRs/branches, or select a review target."""
    if not identifier:
        _list(engine)
        return
    try:
        pr_number = int(identifier)
        _review_pr(engine, pr_number)
        return
    except ValueError:
        pass
    _review_branch(engine, identifier)


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
            head_branch=pr.head_branch,
            updated_at=pr.updated_at,
        )
        _print_review_target(engine)
    except GithubException as e:
        engine._clear()
        engine._warn(f"Could not fetch PR #{pr_number}: {e.data}")
    except RbtrError as e:
        engine._clear()
        engine._warn(str(e))


def _review_branch(engine: Engine, name: str) -> None:
    if engine.session.repo is None:
        return
    for branch_name in engine.session.repo.branches.local:
        if branch_name == name:
            branch = engine.session.repo.branches.local[branch_name]
            commit = branch.peel(pygit2.Commit)
            engine.session.review_target = BranchTarget(
                head_branch=name,
                updated_at=datetime.fromtimestamp(commit.commit_time, tz=UTC),
            )
            _print_review_target(engine)
            return
    engine._warn(f"'{name}' not found as a PR number or local branch.")


def _print_review_target(engine: Engine) -> None:
    target = engine.session.review_target
    if target is None:
        engine._out("No review target selected. Use /review to select one.")
        return
    match target:
        case PRTarget(number=number, title=title, head_branch=branch):
            engine._out(f"Reviewing PR #{number}: {title} ({branch})")
        case BranchTarget(head_branch=branch):
            engine._out(f"Reviewing branch: {branch}")


def _warn_access(engine: Engine, exc: GithubException) -> None:
    engine._warn(
        f"Cannot access {engine.session.owner}/{engine.session.repo_name} "
        f"via GitHub API ({exc.status})."
    )
    message = exc.data.get("message", "") if isinstance(exc.data, dict) else ""
    if message:
        engine._out(message)
    engine._out("Falling back to local branches.")
