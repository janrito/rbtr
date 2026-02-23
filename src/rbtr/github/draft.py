"""Review draft persistence — ``.rbtr/<prefix>DRAFT-<pr>.toml``.

The draft file is the local source of truth while building a review.
It survives agent crashes and session restarts.  Every mutation
persists immediately.

The filename prefix comes from ``config.tools.workspace_prefix``
(default ``REVIEW-``), so the default draft file is
``.rbtr/REVIEW-DRAFT-42.toml``.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import tomli_w

from rbtr.config import WORKSPACE_DIR, config
from rbtr.exceptions import RbtrError
from rbtr.models import InlineComment, ReviewDraft

log = logging.getLogger(__name__)


def _draft_path(pr_number: int) -> Path:
    """Return the path for a PR's draft file."""
    prefix = config.tools.workspace_prefix
    return WORKSPACE_DIR / f"{prefix}DRAFT-{pr_number}.toml"


def load_draft(pr_number: int) -> ReviewDraft | None:
    """Load a draft from disk, or ``None`` if it doesn't exist.

    Raises :class:`~rbtr.exceptions.RbtrError` if the file exists
    but cannot be parsed, so the caller can surface a clear message
    instead of a cryptic traceback.
    """
    path = _draft_path(pr_number)
    if not path.exists():
        return None
    data = path.read_bytes()
    try:
        return ReviewDraft.model_validate(tomllib.loads(data.decode()))
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        log.warning("Corrupt draft %s: %s", path, exc)
        raise RbtrError(
            f"Draft file is corrupt ({path.name}): {exc}\n"
            f"Run /draft clear to delete it and start fresh."
        ) from exc


def save_draft(pr_number: int, draft: ReviewDraft) -> None:
    """Persist *draft* to disk.  Creates parent dirs if needed.

    Validates the serialised TOML round-trips before writing so a
    corrupt file is never persisted.
    """
    path = _draft_path(pr_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = draft.model_dump(
        mode="json", exclude_defaults=True, exclude_unset=True, exclude_none=True
    )
    text = tomli_w.dumps(payload, multiline_strings=True)
    # Guard: verify the TOML we're about to write can be parsed back.
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        log.warning("tomli_w produced unparseable TOML for PR #%d, falling back", pr_number)
        text = tomli_w.dumps(payload)
    path.write_text(text)


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
