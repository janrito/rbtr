"""GitHub API operations for rbtr."""

from datetime import UTC, datetime

from github import Github, GithubException

from rbtr import RbtrError
from rbtr.config import config
from rbtr.models import BranchSummary, PRSummary


def list_open_prs(gh: Github, owner: str, repo_name: str) -> list[PRSummary]:
    """List open pull requests, most recently updated first."""
    repo = gh.get_repo(f"{owner}/{repo_name}")
    pulls = repo.get_pulls(state="open", sort="updated", direction="desc")

    results: list[PRSummary] = []
    for pr in pulls:
        results.append(
            PRSummary(
                number=pr.number,
                title=pr.title,
                author=pr.user.login if pr.user else "unknown",
                head_branch=pr.head.ref,
                updated_at=pr.updated_at or datetime.now(tz=UTC),
            )
        )
    return results


def list_unmerged_branches(
    gh: Github, owner: str, repo_name: str, open_pr_branches: set[str]
) -> list[BranchSummary]:
    """List remote branches that have no open PR, excluding the default branch.

    Returns at most config.github.max_branches results, sorted by most recently updated first.
    """
    repo = gh.get_repo(f"{owner}/{repo_name}")
    default_branch = repo.default_branch

    results: list[BranchSummary] = []
    for branch in repo.get_branches():
        if len(results) >= config.github.max_branches:
            break
        if branch.name == default_branch:
            continue
        if branch.name in open_pr_branches:
            continue
        commit = branch.commit
        results.append(
            BranchSummary(
                name=branch.name,
                last_commit_sha=commit.sha,
                last_commit_message=(
                    commit.commit.message.split("\n", 1)[0] if commit.commit else ""
                ),
                updated_at=commit.commit.committer.date
                if commit.commit and commit.commit.committer
                else datetime.now(tz=UTC),
            )
        )

    results.sort(key=lambda b: b.updated_at, reverse=True)
    return results


def validate_pr_number(gh: Github, owner: str, repo_name: str, pr_number: int) -> PRSummary:
    """Fetch a specific PR by number. Raises RbtrError if not found."""
    repo = gh.get_repo(f"{owner}/{repo_name}")
    try:
        pr = repo.get_pull(pr_number)
    except GithubException as err:
        raise RbtrError(f"PR #{pr_number} not found in {owner}/{repo_name}.") from err
    return PRSummary(
        number=pr.number,
        title=pr.title,
        author=pr.user.login if pr.user else "unknown",
        head_branch=pr.head.ref,
        updated_at=pr.updated_at or datetime.now(tz=UTC),
    )
