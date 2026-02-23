"""Review draft persistence — ``.rbtr/<prefix>DRAFT-<pr>.toml``.

The draft file is the local source of truth while building a review.
It survives agent crashes and session restarts.  Every mutation
persists immediately.

The filename prefix comes from ``config.tools.workspace_prefix``
(default ``REVIEW-``), so the default draft file is
``.rbtr/REVIEW-DRAFT-42.toml``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w

from rbtr.config import WORKSPACE_DIR, config
from rbtr.models import InlineComment, ReviewDraft


def _draft_path(pr_number: int) -> Path:
    """Return the path for a PR's draft file."""
    prefix = config.tools.workspace_prefix
    return WORKSPACE_DIR / f"{prefix}DRAFT-{pr_number}.toml"


def load_draft(pr_number: int) -> ReviewDraft | None:
    """Load a draft from disk, or ``None`` if it doesn't exist."""
    path = _draft_path(pr_number)
    if not path.exists():
        return None
    data = path.read_bytes()
    return ReviewDraft.model_validate(tomllib.loads(data.decode()))


def save_draft(pr_number: int, draft: ReviewDraft) -> None:
    """Persist *draft* to disk.  Creates parent dirs if needed."""
    path = _draft_path(pr_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(draft.model_dump(mode="json")))


def delete_draft(pr_number: int) -> bool:
    """Delete the draft file.  Returns True if it existed."""
    path = _draft_path(pr_number)
    if path.exists():
        path.unlink()
        return True
    return False


def get_unsynced_comments(
    local: ReviewDraft,
    remote_comments: list[InlineComment],
) -> list[InlineComment]:
    """Return remote comments that are not in the local draft.

    Matches by ``(path, line)`` — same logic as ``merge_remote``.
    """
    local_keys = {(c.path, c.line) for c in local.comments}
    return [c for c in remote_comments if (c.path, c.line) not in local_keys]


def merge_remote(
    local: ReviewDraft | None,
    remote_comments: list[InlineComment],
) -> ReviewDraft:
    """Merge remote comments into a local draft.

    Appends remote comments whose ``(path, line)`` pair is not
    already present in the local draft.  Local comments are never
    modified — the local draft is the source of truth for edits.

    If *local* is ``None``, creates a new draft from the remote
    comments.
    """
    if local is None:
        return ReviewDraft(comments=list(remote_comments))

    existing = {(c.path, c.line) for c in local.comments}
    new = [c for c in remote_comments if (c.path, c.line) not in existing]
    if not new:
        return local

    return local.model_copy(update={"comments": [*local.comments, *new]})
