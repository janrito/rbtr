"""Git operations — all pygit2 interaction lives here.

Submodules:

- ``repo``    — repository lifecycle (open, status, branches, remotes, fetch).
- ``objects`` — object-store reading (resolve refs, walk trees, diffs, logs).
- ``filters`` — path filtering (gitignore, globs, binary detection).
"""

from rbtr.git.filters import is_binary, is_path_ignored
from rbtr.git.objects import (
    FileEntry,
    changed_files,
    commit_log_between,
    diff_refs,
    diff_single,
    list_files,
    read_blob,
    resolve_commit,
    walk_tree,
)
from rbtr.git.repo import (
    default_branch,
    fetch_pr_head,
    list_local_branches,
    open_repo,
    parse_github_remote,
    require_clean,
)

__all__ = [
    "FileEntry",
    "changed_files",
    "commit_log_between",
    "default_branch",
    "diff_refs",
    "diff_single",
    "fetch_pr_head",
    "is_binary",
    "is_path_ignored",
    "list_files",
    "list_local_branches",
    "open_repo",
    "parse_github_remote",
    "read_blob",
    "require_clean",
    "resolve_commit",
    "walk_tree",
]
