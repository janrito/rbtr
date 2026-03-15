"""Git operations — all pygit2 interaction lives here.

Submodules:

- `repo`    — repository lifecycle (open, status, branches, remotes, fetch).
- `objects` — object-store reading (resolve refs, walk trees, diffs, logs).
- `filters` — path filtering (gitignore, globs, binary detection).

Only symbols that are used by multiple external modules are
re-exported here.  Specialised types (`DiffResult`, `DiffStats`,
`LogEntry`) and single-caller functions (`diff_refs`,
`diff_single`, `commit_log_between`, `read_blob`) should be
imported from the submodule directly.
"""

from rbtr.git.filters import is_binary, is_path_ignored
from rbtr.git.objects import (
    FileEntry,
    changed_files,
    list_files,
    resolve_commit,
    walk_tree,
)
from rbtr.git.repo import (
    default_branch,
    fetch_pr_refs,
    find_git_root,
    list_local_branches,
    open_repo,
)

__all__ = [
    "FileEntry",
    "changed_files",
    "default_branch",
    "fetch_pr_refs",
    "find_git_root",
    "is_binary",
    "is_path_ignored",
    "list_files",
    "list_local_branches",
    "open_repo",
    "resolve_commit",
    "walk_tree",
]
