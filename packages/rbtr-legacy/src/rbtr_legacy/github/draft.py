"""Review draft persistence and sync matching.

The draft file (`.rbtr/drafts/<pr>.yaml`) is the local source
of truth while building a review.  Every mutation persists
immediately.

Matching
--------
When syncing with GitHub, we need to match remote comments (which
have `github_id`) to local comments.  Two tiers:

**Tier 1 — `github_id`:** If a local comment already has a
`github_id` from a previous sync, match by exact ID.

**Tier 2 — `(path, line, formatted_body)`:** For comments
without a `github_id` (locally new), match by content against
unmatched remote comments.  Pairs greedily, 1:1.

All remaining unmatched remote comments are imported as new.

Dirty detection uses content hashes stored directly on each
comment (`comment_hash`) and on the draft (`summary_hash`).
No separate sync section — each entity tracks its own state.

Tombstones
----------
When a synced comment (has `github_id`) is removed locally,
it becomes a *tombstone*: `body` and `suggestion` are
cleared but the `github_id` and `comment_hash` are
preserved.  See `is_tombstone()`.

During sync, tombstones are:

1. **Matched** to their remote counterpart by `github_id`
   (tier 1).  Three-way merge sees local as dirty → keeps
   the tombstone.
2. **Excluded** from the pushed comments, so the remote
   review is recreated without them.
3. **Dropped** from the saved draft after the push.

This prevents re-import of a comment the user deleted
locally.  Comments without a `github_id` (never synced)
are removed immediately — no tombstone needed.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from rbtr_legacy.config import config
from rbtr_legacy.exceptions import RbtrError
from rbtr_legacy.git.objects import DiffLineRanges, nearest_commentable_line
from rbtr_legacy.github.client import format_comment_body
from rbtr_legacy.models import DiffSide, InlineComment, ReviewDraft
from rbtr_legacy.workspace import resolve_path

log = logging.getLogger(__name__)


# ── Content hashing ──────────────────────────────────────────────────


def _comment_hash(c: InlineComment) -> str:
    """Deterministic hash of a comment's content fields.

    Covers `path`, `line`, `body`, and `suggestion`.
    `side` and `commit_id` are excluded — they are resolution
    metadata not visible in the GitHub UI.

    `line` IS included because the `position` ↔ `line`
    conversion via `_position_to_line` is deterministic (it
    walks the diff hunk GitHub returns).
    """
    content = f"{c.path}\0{c.line}\0{c.body}\0{c.suggestion}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _summary_hash(summary: str) -> str:
    """Deterministic hash of the review summary."""
    return hashlib.sha256(summary.encode()).hexdigest()[:16]


# ── Draft lock ────────────────────────────────────────────────────────

_draft_lock = threading.Lock()


@contextmanager
def draft_transaction() -> Iterator[None]:
    """Serialize draft file access.

    Wrap the full load → modify → save cycle to prevent
    concurrent tool calls from corrupting the file or
    losing edits.
    """
    with _draft_lock:
        yield


# ── Persistence ──────────────────────────────────────────────────────


def draft_path(pr_number: int) -> Path:
    """Return the path for a PR's draft file."""
    return resolve_path(config.tools.drafts_dir) / f"{pr_number}.yaml"


def _yaml() -> YAML:
    """Return a configured YAML instance."""
    y = YAML()
    y.default_flow_style = False
    return y


def _literalize(obj: Any) -> Any:  # Any: recursive JSON value from model_dump
    """Walk a dict/list and wrap multiline strings as literal block scalars.

    This makes `ruamel.yaml` emit `|-` blocks instead of
    escaped `\\n` sequences — much more readable for markdown
    bodies and code suggestions.
    """
    if isinstance(obj, str):
        return LiteralScalarString(obj) if "\n" in obj else obj
    if isinstance(obj, dict):
        return {k: _literalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_literalize(v) for v in obj]
    return obj


def load_draft(pr_number: int) -> ReviewDraft | None:
    """Load a draft from disk, or `None` if it doesn't exist.

    Raises `RbtrError` if the file exists
    but cannot be parsed, so the caller can surface a clear message
    instead of a cryptic traceback.
    """
    path = draft_path(pr_number)
    if not path.exists():
        return None
    try:
        data = _yaml().load(path)
        return ReviewDraft.model_validate(data)
    except Exception as exc:
        log.warning("Corrupt draft %s: %s", path, exc)
        raise RbtrError(
            f"Draft file is corrupt ({path.name}): {exc}\n"
            f"Run /draft clear to delete it and start fresh."
        ) from exc


def save_draft(pr_number: int, draft: ReviewDraft) -> None:
    """Persist *draft* to disk.  Creates parent dirs if needed.

    All non-`None` fields are serialised explicitly — no
    `exclude_defaults` or `exclude_unset`, which are fragile
    under `model_copy` roundtrips.  Validates the serialised
    YAML before writing and uses an atomic temp-file-plus-rename
    so concurrent readers never see a partial write.
    """
    path = draft_path(pr_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = draft.model_dump(mode="json", exclude_none=True)
    payload = _literalize(payload)
    buf = StringIO()
    _yaml().dump(payload, buf)
    text = buf.getvalue()
    # Validate before touching the file.
    try:
        data = _yaml().load(text)
        ReviewDraft.model_validate(data)
    except Exception as exc:
        log.error("Draft roundtrip validation failed for %s: %s", path.name, exc)
        raise RbtrError(
            f"Draft serialisation produced invalid YAML ({path.name}): {exc}\n"
            f"This is a bug — please report it."
        ) from exc
    # Atomic write: temp file in same directory + rename.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def delete_draft(pr_number: int) -> bool:
    """Delete the draft file.  Returns True if it existed."""
    path = draft_path(pr_number)
    if path.exists():
        path.unlink()
        return True
    return False


# ── Matching ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of matching remote comments against a local draft.

    After applying this result the local draft contains the merged
    state of both sides.
    """

    comments: list[InlineComment]
    """Merged comment list — updated locals + imported remotes."""

    warnings: list[str]
    """Human-readable messages about conflicts and remote deletions."""


def match_comments(
    local_comments: list[InlineComment],
    remote_comments: list[InlineComment],
) -> MatchResult:
    """Match remote comments to local and produce a merged list.

    See module docstring for the matching algorithm.
    """
    warnings: list[str] = []

    # Index remote comments by github_id for O(1) lookup.
    remote_by_id: dict[int, InlineComment] = {}
    for rc in remote_comments:
        if rc.github_id is not None:
            remote_by_id[rc.github_id] = rc
    unmatched_remote_ids = set(remote_by_id.keys())

    # Phase 1: process local comments.
    merged: list[InlineComment] = []
    for lc in local_comments:
        if lc.github_id is not None and lc.github_id in remote_by_id:
            # ── Tier 1: matched by github_id ─────────────────────
            rc = remote_by_id[lc.github_id]
            unmatched_remote_ids.discard(lc.github_id)
            updated = _reconcile(lc, rc, warnings)
            merged.append(updated)

        elif lc.github_id is not None and lc.comment_hash:
            # Had a github_id from a previous sync but the remote
            # comment is gone → deleted on GitHub.
            warnings.append(f"Comment on {lc.path}:{lc.line} was deleted on GitHub.")
            # Don't include in merged — honour the remote deletion.

        else:
            # ── No github_id or not synced: locally new. ─────────
            # Try tier 2: content match against unmatched remotes.
            matched_id = _tier2_match(lc, remote_by_id, unmatched_remote_ids)
            if matched_id is not None:
                unmatched_remote_ids.discard(matched_id)
                # Adopt the github_id from the remote match.
                merged.append(lc.model_copy(update={"github_id": matched_id}))
            else:
                # Genuinely new local comment — keep as-is.
                merged.append(lc)

    # Phase 2: import remaining unmatched remote comments.
    for gid in sorted(unmatched_remote_ids):
        rc = remote_by_id[gid]
        merged.append(rc)
        warnings.append(f"New remote comment imported: {rc.path}:{rc.line}")

    return MatchResult(comments=merged, warnings=warnings)


def _reconcile(
    local: InlineComment,
    remote: InlineComment,
    warnings: list[str],
) -> InlineComment:
    """Reconcile a matched local/remote pair using three-way merge.

    The comment's `comment_hash` is the common ancestor.
    """
    comment_hash = local.comment_hash
    if not comment_hash:
        # No merge base — first time seeing this pair.
        # Keep local (it was already present before sync tracking).
        return local

    local_dirty = _comment_hash(local) != comment_hash
    remote_dirty = _comment_hash(remote) != comment_hash

    if not remote_dirty:
        # Remote unchanged — keep whatever local has (maybe edited).
        return local

    if not local_dirty:
        # Only remote changed — accept remote edit.
        return remote

    # Both sides changed — conflict.  Keep local, warn user.
    remote_preview = remote.body[:60]
    warnings.append(
        f"Conflict on {local.path}:{local.line} — keeping local. Remote was: {remote_preview}"
    )
    return local


def _tier2_match(
    local: InlineComment,
    remote_by_id: dict[int, InlineComment],
    unmatched_ids: set[int],
) -> int | None:
    """Try to match a local comment by `(path, line, formatted_body)`.

    Returns the `github_id` of the matched remote, or `None`.
    Only matches if exactly one candidate exists.
    """
    local_key = (local.path, local.line, format_comment_body(local))
    candidates: list[int] = []
    for gid in unmatched_ids:
        rc = remote_by_id[gid]
        remote_key = (rc.path, rc.line, format_comment_body(rc))
        if local_key == remote_key:
            candidates.append(gid)
    if len(candidates) == 1:
        return candidates[0]
    return None


def stamp_synced(draft: ReviewDraft) -> ReviewDraft:
    """Set `comment_hash` on every comment and `summary_hash` on the draft.

    Called after a successful push to record what was sent to GitHub.
    """
    stamped = [c.model_copy(update={"comment_hash": _comment_hash(c)}) for c in draft.comments]
    return draft.model_copy(
        update={
            "comments": stamped,
            "summary_hash": _summary_hash(draft.summary),
        }
    )


def is_tombstone(comment: InlineComment) -> bool:
    """True if *comment* is a local deletion marker.

    A tombstone is a previously-synced comment (has `github_id`)
    whose `body` has been cleared.  It stays in the draft so
    the push excludes it (preventing re-import), then gets dropped
    after a successful sync.
    """
    return comment.github_id is not None and not comment.body


def comment_sync_status(comment: InlineComment) -> str:
    """Return a single-char status indicator for display.

    `✗` tombstone (pending deletion), `★` new (never synced),
    `✎` dirty (modified since sync), `✓` clean (matches synced hash).
    """
    if is_tombstone(comment):
        return "✗"
    if not comment.comment_hash:
        return "★"
    if _comment_hash(comment) != comment.comment_hash:
        return "✎"
    return "✓"


# ── Comment validation ───────────────────────────────────────────────


def find_comment(
    comments: list[InlineComment],
    path: str,
    body_substring: str,
) -> tuple[int, InlineComment] | str:
    """Find a single comment by *path* and *body_substring*.

    Searches *comments* for entries whose `path` matches and
    whose `body` contains *body_substring*.

    Returns `(index, comment)` on a unique match, or an error
    string when zero or multiple comments match.
    """
    matches: list[tuple[int, InlineComment]] = [
        (i, c) for i, c in enumerate(comments) if c.path == path and body_substring in c.body
    ]
    if not matches:
        on_file = [c for c in comments if c.path == path]
        if not on_file:
            return f"No comments on '{path}'."
        return (
            f"No comment on '{path}' contains \"{body_substring}\". "
            f"Comments on this file:\n" + "\n".join(f"  - L{c.line}: {c.body}" for c in on_file)
        )
    if len(matches) > 1:
        return (
            f"\"{body_substring}\" matches {len(matches)} comments on '{path}'. "
            f"Use a more specific substring:\n"
            + "\n".join(f"  - L{c.line}: {c.body}" for _, c in matches)
        )
    return matches[0]


def snap_to_commentable_line(
    ranges: DiffLineRanges,
    path: str,
    line: int,
) -> tuple[int, str | None, str]:
    """Ensure *line* is commentable, snapping to the nearest valid line if not.

    Returns `(line, error, note)` where:

    - *line* is the (possibly adjusted) commentable line.
    - *error* is set when the path is not in the diff (caller
      should reject).
    - *note* describes the snap when the line was adjusted.
    """
    if not ranges:
        return line, None, ""  # no diff data — skip validation
    if path not in ranges:
        available = sorted(ranges.keys())
        return (
            line,
            (
                f"'{path}' is not in the PR diff. "
                f"Comments must target changed files. Changed files: {', '.join(available[:20])}"
            ),
            "",
        )
    if line in ranges[path]:
        return line, None, ""
    # path is in ranges (checked above) → nearest is always valid.
    nearest = nearest_commentable_line(ranges, path, line)
    if nearest is None:  # pragma: no cover — unreachable, path has lines
        return line, None, ""
    return nearest, None, f"Snapped from line {line} to nearest commentable line {nearest}."


def partition_comments(
    comments: list[InlineComment],
    right_ranges: DiffLineRanges,
    left_ranges: DiffLineRanges,
) -> tuple[list[InlineComment], list[InlineComment]]:
    """Split comments into `(valid, invalid)` based on side-aware ranges.

    File-level comments (`line == 0`) are always valid.
    `RIGHT` comments validate against HEAD-side ranges,
    `LEFT` comments validate against BASE-side ranges.
    """
    valid: list[InlineComment] = []
    invalid: list[InlineComment] = []
    for c in comments:
        if c.line == 0:
            valid.append(c)
            continue
        ranges = left_ranges if c.side is DiffSide.LEFT else right_ranges
        if c.path in ranges and c.line in ranges[c.path]:
            valid.append(c)
        else:
            invalid.append(c)
    return valid, invalid
