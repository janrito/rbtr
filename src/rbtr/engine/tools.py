"""LLM tools — expose the code index and git history to the agent.

Each tool is registered on the shared ``agent`` instance via
``@agent.tool``.  Tools receive ``RunContext[AgentDeps]`` and
read ``state.index`` / ``state.review_target`` / ``state.repo``.

Index tools are hidden when no index is loaded.  Git tools are
hidden when no repo is available.  File tools need only a repo.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pygit2
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from rbtr.config import config
from rbtr.git import is_binary, is_path_ignored, resolve_commit, walk_tree
from rbtr.git.objects import (
    DiffLineRanges,
    DiffResult,
    DiffStats,
    commit_log_between,
    diff_line_ranges,
    diff_line_ranges_left,
    diff_refs,
    diff_single,
    read_blob,
    resolve_anchor,
)
from rbtr.github.client import get_pr_discussion as fetch_pr_discussion
from rbtr.github.draft import load_draft, save_draft
from rbtr.index.models import ChunkKind, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.models import (
    DiscussionEntry,
    DiscussionEntryKind,
    InlineComment,
    PRTarget,
    ReviewDraft,
)

from .agent import AgentDeps, agent

# ── Prepare functions ────────────────────────────────────────────────


async def _require_index(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when an index is available."""
    if ctx.deps.state.index is None or ctx.deps.state.review_target is None:
        return None
    return tool_def


async def _require_repo(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a repo + review target is available."""
    if ctx.deps.state.repo is None or ctx.deps.state.review_target is None:
        return None
    return tool_def


async def _require_pr(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a PR target and GitHub auth are available."""
    state = ctx.deps.state
    if state.gh is None or not isinstance(state.review_target, PRTarget):
        return None
    return tool_def


async def _require_pr_target(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a PR target is selected.

    Unlike ``_require_pr``, does not require GitHub auth — draft
    management is purely local.
    """
    if not isinstance(ctx.deps.state.review_target, PRTarget):
        return None
    return tool_def


# ── Accessor helpers ─────────────────────────────────────────────────


def _head_ref(ctx: RunContext[AgentDeps]) -> str:
    """Return the git-resolvable head ref from the review target."""
    target = ctx.deps.state.review_target
    if target is None:  # pragma: no cover — guarded by prepare
        msg = "no review target"
        raise RuntimeError(msg)
    return target.head_ref


def _store(ctx: RunContext[AgentDeps]) -> IndexStore:
    """Return the index store."""
    store = ctx.deps.state.index
    if store is None:  # pragma: no cover — guarded by _require_index
        msg = "no index store"
        raise RuntimeError(msg)
    return store


def _repo(ctx: RunContext[AgentDeps]) -> pygit2.Repository:
    """Return the git repo."""
    repo = ctx.deps.state.repo
    if repo is None:  # pragma: no cover — guarded by _require_repo
        msg = "no repository"
        raise RuntimeError(msg)
    return repo


def _resolve_tool_ref(ctx: RunContext[AgentDeps], ref: str) -> str:
    """Map ``"head"`` / ``"base"`` to the review target's branch names.

    Any other value is returned as-is (raw git ref).
    """
    target = ctx.deps.state.review_target
    match ref:
        case "head":
            if target is None:  # pragma: no cover — guarded by prepare
                msg = "no review target"
                raise RuntimeError(msg)
            return target.head_ref
        case "base":
            if target is None:  # pragma: no cover — guarded by prepare
                msg = "no review target"
                raise RuntimeError(msg)
            return target.base_branch
        case _:
            return ref


# ── Output limiting ──────────────────────────────────────────────────


def _limited(shown: int, total: int, *, hint: str) -> str:
    """Standard truncation trailer appended when output is capped.

    Every tool that caps output uses this, so the LLM sees a
    consistent format and knows how to request more.
    """
    return f"\n\n... limited ({shown}/{total}). {hint}"


# ── GitHub discussion tool (require PR) ──────────────────────────────


def _format_discussion_entry(entry: DiscussionEntry) -> str:
    """Format a single discussion entry as text for LLM consumption.

    Produces a self-contained block with header (timestamp, author,
    kind-specific metadata), optional diff hunk, body, and reactions.
    """
    ts = entry.created_at.strftime("%Y-%m-%d %H:%M")
    bot_tag = " [bot]" if entry.is_bot else ""
    header_parts: list[str] = [f"[{ts}] @{entry.author}{bot_tag}"]

    match entry.kind:
        case DiscussionEntryKind.REVIEW:
            header_parts.append(f"({entry.review_state})")
        case DiscussionEntryKind.INLINE:
            location = entry.path
            if entry.line is not None:
                location += f":{entry.line}"
            header_parts.append(f"on {location}")
            if entry.in_reply_to_id is not None:
                header_parts.append("(reply)")
        case DiscussionEntryKind.COMMENT:
            pass

    header = " ".join(header_parts)
    parts: list[str] = [header]

    if entry.diff_hunk:
        parts.append(f"```\n{entry.diff_hunk}\n```")

    if entry.body:
        parts.append(entry.body)

    if entry.reactions:
        reaction_str = "  ".join(f"{emoji} {count}" for emoji, count in entry.reactions.items())
        parts.append(f"Reactions: {reaction_str}")

    return "\n".join(parts)


@agent.tool(prepare=_require_pr)
def get_pr_discussion(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Read the existing discussion on the current pull request.

    Fetches all reviews, inline comments, and general comments
    from GitHub, sorted chronologically (oldest first).  Includes
    bot comments (CI, linters) and emoji reactions.

    Use this before starting a review to see what's already been
    said — avoid duplicating existing feedback, build on
    unresolved threads, and respect resolved discussions.

    Each entry shows the author, timestamp, type (review/inline/
    comment), and body.  Inline comments include the file path,
    line number, and diff context.  Reviews show their verdict
    (APPROVED, CHANGES_REQUESTED, etc.).

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.

    Args:
        offset: Number of entries to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum entries to return per call
            (defaults to ``tools.max_results`` config value).
    """
    state = ctx.deps.state
    target = state.review_target
    gh_ctx = state.gh_ctx
    if not isinstance(target, PRTarget) or gh_ctx is None:
        return "No PR selected or not authenticated with GitHub."

    # Fetch and cache on state for the duration of the review.
    if state.discussion_cache is None:
        state.discussion_cache = fetch_pr_discussion(gh_ctx, target.number)
    entries = state.discussion_cache

    if not entries:
        return "No discussion on this PR yet."

    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    total = len(entries)
    page: list[DiscussionEntry] = entries[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} entries."

    lines = [_format_discussion_entry(entry) for entry in page]
    result = "\n\n---\n\n".join(lines)
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


# ── Draft management tools (require PR target) ──────────────────────


def _pr_number(ctx: RunContext[AgentDeps]) -> int:
    """Return the PR number from the review target."""
    target = ctx.deps.state.review_target
    if not isinstance(target, PRTarget):  # pragma: no cover — guarded by prepare
        msg = "no PR target"
        raise RuntimeError(msg)
    return target.number


def _load_or_create_draft(pr_number: int) -> ReviewDraft:
    """Load the draft from disk or create an empty one."""
    return load_draft(pr_number) or ReviewDraft()


# Cached diff line ranges — invalidated on new /review.
_cached_ranges: DiffLineRanges | None = None
_cached_ranges_left: DiffLineRanges | None = None
_cached_ranges_key: tuple[str, str] = ("", "")


def _get_diff_ranges(
    ctx: RunContext[AgentDeps],
    *,
    side: str = "RIGHT",
) -> DiffLineRanges:
    """Return commentable line ranges for the current review target.

    Caches the result so repeated ``add_review_comment`` calls in the
    same turn don't recompute the diff.  *side* selects HEAD
    (``"RIGHT"``) or BASE (``"LEFT"``) ranges.
    """
    global _cached_ranges, _cached_ranges_left, _cached_ranges_key
    state = ctx.deps.state
    target = state.review_target
    repo = state.repo
    if target is None or repo is None:
        return {}
    key = (target.base_branch, target.head_ref)
    if _cached_ranges_key != key:
        _cached_ranges = None
        _cached_ranges_left = None
        _cached_ranges_key = key
    if side == "LEFT":
        if _cached_ranges_left is None:
            try:
                _cached_ranges_left = diff_line_ranges_left(
                    repo, target.base_branch, target.head_ref
                )
            except KeyError:
                _cached_ranges_left = {}
        return _cached_ranges_left
    if _cached_ranges is None:
        try:
            _cached_ranges = diff_line_ranges(repo, target.base_branch, target.head_ref)
        except KeyError:
            _cached_ranges = {}
    return _cached_ranges


def find_comment(
    comments: list[InlineComment],
    path: str,
    body_substring: str,
) -> tuple[int, InlineComment] | str:
    """Find a single comment by *path* and *body_substring*.

    Searches *comments* for entries whose ``path`` matches and
    whose ``body`` contains *body_substring*.

    Returns ``(index, comment)`` on a unique match, or an error
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
            f"Comments on this file:\n" + "\n".join(f"  - {c.body[:80]}" for c in on_file)
        )
    if len(matches) > 1:
        return (
            f"\"{body_substring}\" matches {len(matches)} comments on '{path}'. "
            f"Use a more specific substring:\n"
            + "\n".join(f"  - {c.body[:80]}" for _, c in matches)
        )
    return matches[0]


def _validate_comment_location(
    ranges: DiffLineRanges,
    path: str,
    line: int,
) -> str | None:
    """Return an error message if *path*/*line* can't be commented on, else None."""
    if not ranges:
        return None  # no diff data — skip validation
    if path not in ranges:
        available = sorted(ranges.keys())
        return (
            f"'{path}' is not in the PR diff. "
            f"Comments must target changed files. Changed files: {', '.join(available[:20])}"
        )
    valid_lines = ranges[path]
    if line not in valid_lines:
        # Find nearest valid lines for a helpful message.
        sorted_lines = sorted(valid_lines)
        nearest = min(sorted_lines, key=lambda n: abs(n - line))
        return (
            f"Line {line} in '{path}' is not in the PR diff. "
            f"Use a line that appears in a changed hunk. "
            f"Nearest commentable line: {nearest}."
        )
    return None


@agent.tool(prepare=_require_pr_target)
def add_review_comment(
    ctx: RunContext[AgentDeps],
    path: str,
    anchor: str,
    body: str,
    suggestion: str = "",
    ref: str = "head",
) -> str:
    """Add an inline comment to the review draft.

    Appends a new comment targeting a specific file and code
    location.  The draft is saved to disk immediately.  Use
    this to build up review feedback incrementally.

    The comment will appear on the PR at the anchored location
    when the review is posted via ``/draft post``.

    **Important:** the ``path`` must be a file changed in the PR,
    and the ``anchor`` must resolve to a line visible in a diff
    hunk (a changed line or a nearby context line).  Use the
    ``diff`` tool first to see which code is available.

    Args:
        path: File path relative to repo root
            (e.g. ``src/api/handler.py``).  Must be a file
            that was changed in the PR diff.
        anchor: An exact substring of the file content at
            ``ref``.  Must match exactly one location.  The
            comment is placed on the **last line** of the match.
            Use a short, unique snippet — one or two lines of
            code from the diff output is ideal.
        body: Markdown body of the comment.  Include a severity
            label (e.g. ``**blocker:**``) as part of the text
            when appropriate.
        suggestion: Optional replacement code.  When non-empty,
            posted as a GitHub suggestion block that the author
            can apply directly.
        ref: Which version of the file to resolve the anchor
            against.  ``"head"`` (default) targets the new code.
            ``"base"`` targets the old (deleted/modified) code.
    """
    if ref not in ("head", "base"):
        return 'ref must be "head" or "base".'

    pr = _pr_number(ctx)
    state = ctx.deps.state
    target = state.review_target
    repo = state.repo

    if not isinstance(target, PRTarget):  # pragma: no cover — guarded by prepare
        return "No PR target."
    if repo is None:  # pragma: no cover — guarded by prepare
        return "No repository."

    # Determine side and resolve ref.
    if ref == "head":
        side = "RIGHT"
        resolved_ref = target.head_ref
    else:
        side = "LEFT"
        resolved_ref = target.base_branch

    # Resolve anchor → line number.
    result = resolve_anchor(repo, resolved_ref, path, anchor)
    if isinstance(result, str):
        return f"Cannot add comment: {result}"
    line = result

    # Validate line is in the PR diff.
    ranges = _get_diff_ranges(ctx, side=side)
    error = _validate_comment_location(ranges, path, line)
    if error:
        return (
            "Cannot add comment: the anchored code is not in the PR diff. "
            "Comment on code that was changed or is near a change."
        )

    draft = _load_or_create_draft(pr)

    comment = InlineComment(
        path=path,
        line=line,
        side=side,
        commit_id=target.head_sha,
        body=body,
        suggestion=suggestion,
    )
    draft = draft.model_copy(update={"comments": [*draft.comments, comment]})
    save_draft(pr, draft)

    return f"Comment added ({path}:{line}). Draft has {len(draft.comments)} comment(s)."


@agent.tool(prepare=_require_pr_target)
def edit_review_comment(
    ctx: RunContext[AgentDeps],
    path: str,
    comment: str,
    body: str = "",
    suggestion: str | None = None,
) -> str:
    """Edit an existing comment in the review draft.

    Locate a comment by file path and a substring of its body,
    then update its body or suggestion.  Only the fields you
    provide are changed — omit a field to keep its current value.

    Args:
        path: File path the comment belongs to
            (e.g. ``src/api/handler.py``).
        comment: A substring that uniquely identifies the
            comment's body on this file.  Quote a distinctive
            phrase from your earlier comment.
        body: New markdown body.  Empty string keeps the
            current body.
        suggestion: New replacement code.  ``None`` keeps the
            current value.  Empty string clears it.
    """

    pr = _pr_number(ctx)
    draft = _load_or_create_draft(pr)

    if not draft.comments:
        return "Draft has no comments to edit."

    result = find_comment(draft.comments, path, comment)
    if isinstance(result, str):
        return f"Cannot edit comment: {result}"

    index, matched = result
    updates: dict[str, str] = {}

    if body:
        updates["body"] = body
    if suggestion is not None:
        updates["suggestion"] = suggestion

    updated = matched.model_copy(update=updates)
    new_comments = list(draft.comments)
    new_comments[index] = updated
    draft = draft.model_copy(update={"comments": new_comments})
    save_draft(pr, draft)

    return f"Comment updated ({updated.path}:{updated.line})."


@agent.tool(prepare=_require_pr_target)
def remove_review_comment(
    ctx: RunContext[AgentDeps],
    path: str,
    comment: str,
) -> str:
    """Remove a comment from the review draft.

    Locate a comment by file path and a substring of its body,
    then remove it from the draft.

    Args:
        path: File path the comment belongs to
            (e.g. ``src/api/handler.py``).
        comment: A substring that uniquely identifies the
            comment's body on this file.  Quote a distinctive
            phrase from your earlier comment.
    """

    pr = _pr_number(ctx)
    draft = _load_or_create_draft(pr)

    if not draft.comments:
        return "Draft has no comments to remove."

    result = find_comment(draft.comments, path, comment)
    if isinstance(result, str):
        return f"Cannot remove comment: {result}"

    index, removed = result

    if removed.github_id is not None:
        # Synced comment — tombstone it so the next push excludes
        # it and the remote copy is not re-imported on pull.
        tombstoned = removed.model_copy(update={"body": "", "suggestion": ""})
        new_comments = list(draft.comments)
        new_comments[index] = tombstoned
    else:
        # Never synced — safe to drop entirely.
        new_comments = [c for i, c in enumerate(draft.comments) if i != index]

    draft = draft.model_copy(update={"comments": new_comments})
    save_draft(pr, draft)

    return (
        f"Removed comment ({removed.path}:{removed.line}). "
        f"Draft has {len(new_comments)} comment(s)."
    )


@agent.tool(prepare=_require_pr_target)
def set_review_summary(
    ctx: RunContext[AgentDeps],
    summary: str,
) -> str:
    """Set or replace the top-level summary of the review draft.

    This is the main body of the review that appears at the top
    when posted to GitHub — a high-level assessment of the PR.

    Args:
        summary: Markdown text for the review summary.
    """

    pr = _pr_number(ctx)
    draft = _load_or_create_draft(pr)
    draft = draft.model_copy(update={"summary": summary})
    save_draft(pr, draft)

    return f"Review summary updated ({len(summary)} chars)."


# ── Search tools (require index) ────────────────────────────────────


@agent.tool(prepare=_require_index)
def search_symbols(
    ctx: RunContext[AgentDeps],
    name: str,
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Search for functions, classes, and methods by name substring.

    Looks up the code index for symbols whose name contains the
    given substring (case-insensitive).  Returns up to
    ``max_results`` matches with their kind, scope, file path,
    and line number.

    Use short, simple names — NOT fully-qualified module paths.
    For example, use ``MQ`` not ``lib.mq.MQ``, or ``crawl`` not
    ``crawler.module.crawl``.

    Use this when you know (part of) a symbol's name and want to
    locate it.  For keyword search across full source content, use
    ``search_codebase`` instead.  To read the source once found,
    use ``read_symbol``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        name: Short substring to match against symbol names
            (e.g. ``parse``, ``Client``, ``MQ``).  Case-insensitive.
        offset: Number of results to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum results to return per call
            (defaults to ``tools.max_results`` config value).
    """
    store = _store(ctx)
    chunks = store.search_by_name(_head_ref(ctx), name)
    if not chunks:
        return f"No symbols matching '{name}'. Try a shorter or different substring."
    total = len(chunks)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    page = chunks[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} results."
    lines: list[str] = []
    for c in page:
        scope = f"{c.scope}." if c.scope else ""
        lines.append(f"{c.kind} {scope}{c.name}  ({c.file_path}:{c.line_start})")
    result = "\n".join(lines)
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


@agent.tool(prepare=_require_index)
def search_codebase(
    ctx: RunContext[AgentDeps],
    query: str,
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Full-text keyword search across the indexed codebase using BM25 ranking.

    Searches both symbol names and their full source content.
    Results are ranked by relevance — symbols that mention the
    keywords more frequently and prominently score higher.
    Returns up to ``max_results`` results with relevance scores,
    kind, scope, file path, and line number.

    Use this when you want to find code that *mentions* certain
    terms but you don't know the symbol name.  For example,
    searching ``"pool_size"`` finds all symbols that reference
    that config key.  For name-only lookup, use
    ``search_symbols``.  For conceptual/semantic matching (no
    shared keywords), use ``search_similar``.

    Only searches indexed code files — config, lockfiles, and
    other non-indexed files are not covered.  Use ``grep``
    for raw text search in specific files.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        query: Keywords to search for (e.g. ``retry timeout``,
            ``database connection``).  Multiple words are treated
            as a combined query.
        offset: Number of results to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum results to return per call
            (defaults to ``tools.max_results`` config value).
    """
    store = _store(ctx)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    # Fetch extra to detect if more exist beyond the page.
    results = store.search_fulltext(_head_ref(ctx), query, top_k=offset + limit + 1)
    if not results:
        return f"No results for '{query}'."
    total = len(results)
    page = results[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} results."
    lines: list[str] = []
    for chunk, score in page:
        scope = f"{chunk.scope}." if chunk.scope else ""
        lines.append(
            f"[{score:.2f}] {chunk.kind} {scope}{chunk.name}"
            f"  ({chunk.file_path}:{chunk.line_start})"
        )
    result = "\n".join(lines)
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


@agent.tool(prepare=_require_index)
def search_similar(
    ctx: RunContext[AgentDeps],
    query: str,
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Semantic similarity search — find code conceptually related to a query.

    Embeds the query into a vector and finds symbols whose
    embedding is closest by cosine similarity.  This finds
    conceptually related code even when it shares no keywords
    with the query.

    Returns up to ``max_results`` results with similarity scores
    (0.0 to 1.0), kind, scope, file path, and line number.

    Use this when you know *what* you're looking for in plain
    language but not the exact names or terms used in the code.
    For keyword matching, use ``search_codebase``.  For name
    lookup, use ``search_symbols``.

    May be unavailable if the embedding model is not loaded —
    falls back to a message suggesting keyword alternatives.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        query: Natural-language description of what you're
            looking for (e.g. ``"rate limiting logic"``,
            ``"database connection pooling"``).
        offset: Number of results to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum results to return per call
            (defaults to ``tools.max_results`` config value).
    """
    store = _store(ctx)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    try:
        results = store.search_by_text(_head_ref(ctx), query, top_k=offset + limit + 1)
    except Exception:
        return (
            "Semantic search unavailable (embedding model not loaded). "
            "Use search_codebase (keyword search) or search_symbols instead."
        )
    if not results:
        return f"No similar symbols for '{query}'."
    total = len(results)
    page = results[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} results."
    lines: list[str] = []
    for chunk, score in page:
        scope = f"{chunk.scope}." if chunk.scope else ""
        lines.append(
            f"[{score:.3f}] {chunk.kind} {scope}{chunk.name}"
            f"  ({chunk.file_path}:{chunk.line_start})"
        )
    result = "\n".join(lines)
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


# ── Index read tools (require index) ─────────────────────────────────


@agent.tool(prepare=_require_index)
def read_symbol(
    ctx: RunContext[AgentDeps],
    name: str,
    ref: str = "head",
) -> str:
    """Read the full source code of a symbol from the index.

    Looks up the symbol by name in the code index and returns
    its source code, preceded by a header showing the symbol
    kind, scope, file path, and line range.  Output is capped
    at ``max_lines``.  Prefers code symbols (functions, classes,
    methods) over other chunk types like tests or doc sections.

    Use this when you've identified a symbol (via
    ``search_symbols``, ``find_references``, etc.) and want to
    read its implementation.  For reading arbitrary file content
    (including non-indexed files), use ``read_file``.

    Use a short name — NOT a fully-qualified path.  For example,
    ``MQ`` not ``lib.mq.MQ``.  If unsure of the exact name, use
    ``search_symbols`` first.

    Args:
        name: Short name to match (e.g. ``MQ``,
            ``parse_config``).  Case-insensitive substring match.
            Returns the first matching code symbol.
        ref: Which version of the codebase to read — ``"head"``
            (default), ``"base"``, or a raw commit SHA.  Returns
            the state at that snapshot, not changes introduced
            by it.
    """
    store = _store(ctx)
    resolved = _resolve_tool_ref(ctx, ref)
    matches = store.search_by_name(resolved, name)

    code_kinds = frozenset({ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD})
    symbols = [c for c in matches if c.kind in code_kinds]

    if not symbols:
        return f"No symbol matching '{name}'. Use search_symbols with a shorter substring."

    sym = symbols[0]
    header = f"# {sym.kind} {sym.scope + '.' if sym.scope else ''}{sym.name}"
    location = f"# {sym.file_path}:{sym.line_start}-{sym.line_end}"
    content_lines = sym.content.splitlines()
    max_lines = config.tools.max_lines
    if len(content_lines) > max_lines:
        body = "\n".join(content_lines[:max_lines])
        total = len(content_lines)
        return f"{header}\n{location}\n\n{body}" + _limited(
            max_lines,
            total,
            hint=f"use read_file('{sym.file_path}', offset={max_lines}) to continue",
        )
    return f"{header}\n{location}\n\n{sym.content}"


@agent.tool(prepare=_require_index)
def list_symbols(
    ctx: RunContext[AgentDeps],
    path: str,
    ref: str = "head",
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """List the symbols (functions, classes, methods) in a file.

    Returns a structural table of contents for the file: each
    symbol's line number, kind, scope, and name.  Only includes
    code symbols from the index — functions, classes, and methods.

    Parallel to ``list_files``: ``list_files`` lists files in a
    directory, ``list_symbols`` lists symbols in a file.

    Use this for orientation before deciding which symbols to
    read in full with ``read_symbol``.  For reading raw file
    content (including non-indexed files), use ``read_file``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        path: File path relative to repo root
            (e.g. ``src/api/handler.py``).  Must not contain
            ``..``.
        ref: Which version of the codebase to read — ``"head"``
            (default), ``"base"``, or a raw commit SHA.  Returns
            the state at that snapshot, not changes introduced
            by it.
        offset: Number of symbols to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum symbols to return per call
            (defaults to ``tools.max_results`` config value).
    """
    if err := _validate_path(path):
        return err

    store = _store(ctx)
    resolved = _resolve_tool_ref(ctx, ref)
    chunks = store.get_chunks(resolved, file_path=path)
    code_kinds = frozenset({ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD})
    symbols = sorted(
        (c for c in chunks if c.kind in code_kinds),
        key=lambda c: c.line_start,
    )

    if not symbols:
        return f"No symbols found in '{path}' at ref '{ref}'."

    total = len(symbols)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    page = symbols[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} symbols."

    lines = [f"# {path}  ({total} symbols)"]
    for s in page:
        scope = f"{s.scope}." if s.scope else ""
        lines.append(f"  {s.line_start:>4d}  {s.kind} {scope}{s.name}")
    result = "\n".join(lines)
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


# ── Dependency graph tools (require index) ───────────────────────────


@agent.tool(prepare=_require_index)
def find_references(
    ctx: RunContext[AgentDeps],
    name: str,
    kind: str = "",
    ref: str = "head",
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Find symbols that reference a given symbol via the dependency graph.

    Walks all inbound edges pointing at the target symbol and
    returns each referencing symbol's kind, name, file path, and
    line number, labelled by edge type.

    By default returns all edge types.  Use ``kind`` to filter
    to a specific relationship:

    - **imports** — the symbol is imported by the source.
      "Who uses this symbol?"  Example: modules that import
      a class or call a function from another module.
    - **calls** — the symbol is called by the source.
      "Who calls this function?"  Example: functions that
      directly invoke the target function.
    - **inherits** — the symbol is a base class of the source.
      "Who extends this class?"  Example: subclasses that
      inherit from the target class.
    - **tests** — the source is a test for the symbol.
      "What tests cover this?"  Example: test functions that
      exercise the target function.
    - **documents** — the source is documentation for the symbol.
      "What docs describe this?"  Example: doc sections or
      README entries that reference the target.
    - **configures** — the source is configuration for the symbol.
      "What config keys affect this?"  Example: config entries
      that parameterise the target's behaviour.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        name: Short symbol name to look up
            (e.g. ``Config``, ``parse_request``).
            Case-insensitive substring match, same as
            ``search_symbols``.  Use a short name, not a
            fully-qualified module path.
        kind: Edge type to filter by (e.g. ``"imports"``,
            ``"tests"``).  Empty string (default) returns all
            edge types.  Must be one of: ``imports``, ``calls``,
            ``inherits``, ``tests``, ``documents``,
            ``configures``.
        ref: Which version of the codebase to query — ``"head"``
            (default), ``"base"``, or a raw commit SHA.  Returns
            the state at that snapshot, not changes introduced
            by it.
        offset: Number of results to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum results to return per call
            (defaults to ``tools.max_results`` config value).
    """
    store = _store(ctx)
    resolved = _resolve_tool_ref(ctx, ref)

    targets = store.search_by_name(resolved, name)
    if not targets:
        return f"Symbol '{name}' not found."

    target_ids = {c.id for c in targets}

    # Resolve the kind filter.
    edge_kind: EdgeKind | None = None
    if kind:
        try:
            edge_kind = EdgeKind(kind)
        except ValueError:
            valid = ", ".join(e.value for e in EdgeKind)
            return f"Unknown edge kind '{kind}'. Valid kinds: {valid}."

    edges = store.get_edges(resolved)
    matching = [
        e for e in edges if e.target_id in target_ids and (edge_kind is None or e.kind == edge_kind)
    ]

    if not matching:
        if kind:
            return f"No '{kind}' references found for '{name}'."
        return f"No references found for '{name}'."

    all_chunks = {c.id: c for c in store.get_chunks(resolved)}
    all_lines: list[str] = []
    for edge in matching:
        src = all_chunks.get(edge.source_id)
        if src:
            all_lines.append(
                f"[{edge.kind.value}] {src.kind} {src.name}  ({src.file_path}:{src.line_start})"
            )

    if not all_lines:
        return f"No references found for '{name}'."

    total = len(all_lines)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    page = all_lines[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} references."
    result = "\n".join(page)
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


# ── Structural diff (require index) ──────────────────────────────────


@agent.tool(prepare=_require_index)
def changed_symbols(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_lines: int | None = None,
) -> str:
    """List symbols changed between base and head.

    Compares the indexed symbols at the review target's base
    branch against the head branch to identify structural
    changes at the symbol level — not line-by-line text diffs.
    Parallel to ``changed_files`` (file paths) vs
    ``changed_symbols`` (symbols).

    Returns categorised sections (only non-empty sections shown):

    - **Added**: symbols present in head but not in base.
    - **Removed**: symbols in base but not in head.
    - **Modified**: same symbol name/scope but different content.
    - **Stale docs**: doc sections that reference modified symbols
      but were not themselves updated.
    - **Missing tests**: new functions/methods with no TESTS edge.
    - **Broken edges**: import edges in head that point at symbols
      removed in head.

    Use this as a high-level overview to guide the review.  For
    the raw line-by-line diff, use ``diff``.  For the list of
    changed file paths, use ``changed_files``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_lines`` (up to the
    configured cap) to get more per page.

    Args:
        offset: Number of output lines to skip (default 0).
            Use to fetch the next page when a previous call was
            limited.
        max_lines: Maximum lines to return per call
            (defaults to ``tools.max_lines`` config value).
    """
    from rbtr.index.orchestrator import compute_diff  # deferred: avoid circular

    store = _store(ctx)
    target = ctx.deps.state.review_target
    if target is None:
        return "No review target selected."

    sd = compute_diff(target.base_branch, target.head_ref, store)
    sections: list[str] = []

    if sd.added:
        lines = [f"  {c.kind} {c.name}  ({c.file_path}:{c.line_start})" for c in sd.added]
        sections.append(f"Added ({len(sd.added)}):\n" + "\n".join(lines))

    if sd.removed:
        lines = [f"  {c.kind} {c.name}  ({c.file_path}:{c.line_start})" for c in sd.removed]
        sections.append(f"Removed ({len(sd.removed)}):\n" + "\n".join(lines))

    if sd.modified:
        lines = [f"  {c.kind} {c.name}  ({c.file_path}:{c.line_start})" for c in sd.modified]
        sections.append(f"Modified ({len(sd.modified)}):\n" + "\n".join(lines))

    if sd.stale_docs:
        lines = [
            f"  {doc.name} ({doc.file_path}) → {code.name} ({code.file_path})"
            for doc, code in sd.stale_docs
        ]
        sections.append(f"Stale docs ({len(sd.stale_docs)}):\n" + "\n".join(lines))

    if sd.missing_tests:
        lines = [f"  {c.kind} {c.name}  ({c.file_path}:{c.line_start})" for c in sd.missing_tests]
        sections.append(f"Missing tests ({len(sd.missing_tests)}):\n" + "\n".join(lines))

    if sd.broken_edges:
        lines = [f"  {e.source_id} → {e.target_id}  ({e.kind})" for e in sd.broken_edges]
        sections.append(f"Broken edges ({len(sd.broken_edges)}):\n" + "\n".join(lines))

    if not sections:
        return "No structural differences between base and head."

    all_lines = "\n\n".join(sections).splitlines()
    total = len(all_lines)
    limit = (
        min(max_lines, config.tools.max_lines) if max_lines is not None else config.tools.max_lines
    )
    page = all_lines[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} lines."
    result = "\n".join(page)
    if offset + len(page) < total:
        result += _limited(
            offset + len(page), total, hint=f"offset={offset + len(page)} to continue"
        )
    return result


# ── Git tools ────────────────────────────────────────────────────────


@agent.tool(prepare=_require_repo)
def changed_files(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """List file paths changed between the review target's base and head.

    Returns a sorted list of every file that was added, modified,
    or deleted.

    Use this as the starting point for systematic file-by-file
    review: get the list of changed paths, then examine each
    with ``diff(path=...)``, ``read_file``, or ``list_symbols``.
    For a structural summary of *symbol-level* changes, use
    ``changed_symbols``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        offset: Number of entries to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum entries to return per call
            (defaults to ``tools.max_results`` config value).
    """
    from rbtr.git import changed_files as _changed_files  # deferred: avoid circular

    repo = _repo(ctx)
    target = ctx.deps.state.review_target
    if target is None:
        return "No review target selected."

    try:
        paths = _changed_files(repo, target.base_branch, target.head_ref)
    except KeyError as exc:
        return f"Could not resolve refs: {exc}"

    if not paths:
        return "No files changed between base and head."

    sorted_paths = sorted(paths)
    total = len(sorted_paths)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    page = sorted_paths[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} changed files."

    header = f"Changed files ({total}):"
    listing = "\n".join(f"  {p}" for p in page)
    result = f"{header}\n{listing}"
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


@agent.tool(prepare=_require_repo)
def diff(
    ctx: RunContext[AgentDeps],
    path: str = "",
    ref: str = "",
    offset: int = 0,
    max_lines: int | None = None,
) -> str:
    """Show a unified text diff from the git repository.

    Produces standard unified diff output (``---``/``+++`` with
    ``@@`` hunks) showing exact line-level changes.  Output is
    capped at ``max_lines``.

    Three modes depending on ``ref``:

    - **Empty string** (default): diffs the review target's base
      branch against its head branch — the main review diff.
    - **Single ref** (e.g. ``"abc123"``): diffs that commit
      against its parent.
    - **Range** (e.g. ``"main..feature"``): diffs between two
      arbitrary refs.

    When ``path`` is set, only patches for that file are shown.
    Use this for file-by-file review of large PRs instead of
    reading the entire diff at once.

    Use this for the raw line-by-line view of what changed.
    For a structural/symbol-level summary, use ``changed_symbols``.
    For the list of changed paths, use ``changed_files``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_lines`` (up to the
    configured cap) to get more per page.

    Args:
        path: File path to restrict the diff to
            (e.g. ``src/api/handler.py``).  Empty string
            (default) shows the full diff.
        ref: A commit SHA, branch name, or ``base..head`` range.
            Empty string (default) diffs the review target.
        offset: Number of output lines to skip (0-indexed,
            default 0).  Use to paginate large diffs.
        max_lines: Maximum lines of diff output to return
            (defaults to ``tools.max_lines`` config value).
    """
    repo = _repo(ctx)
    target = ctx.deps.state.review_target
    capped = (
        min(max_lines, config.tools.max_lines) if max_lines is not None else config.tools.max_lines
    )

    try:
        if not ref:
            if target is None:
                return "No review target selected."
            result = diff_refs(repo, target.base_branch, target.head_ref, path=path)
        elif ".." in ref:
            parts = ref.split("..", 1)
            result = diff_refs(repo, parts[0], parts[1], path=path)
        else:
            result = diff_single(repo, ref, path=path)
    except (KeyError, ValueError) as exc:
        return exc.args[0] if exc.args else str(exc)

    return _paginate_diff(result, offset, capped)


def _format_diff_stats(s: DiffStats) -> str:
    """Format ``DiffStats`` into a one-line summary."""
    return f"{s.files_changed} files changed, +{s.insertions} -{s.deletions}"


def _paginate_diff(dr: DiffResult, offset: int, max_lines: int) -> str:
    """Apply pagination to a ``DiffResult``."""
    total = len(dr.patch_lines)
    page = dr.patch_lines[offset : offset + max_lines]

    text = _format_diff_stats(dr.stats) + "\n\n" + "\n".join(page)
    shown_end = offset + len(page)
    if shown_end < total:
        text += _limited(shown_end, total, hint=f"offset={shown_end} to continue")
    return text


@agent.tool(prepare=_require_repo)
def commit_log(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Show the commit log between the review target's base and head.

    Lists commits on the head branch that are not on the base
    branch, in reverse chronological order.  Each line shows the
    short SHA, author name, and first line of the commit message.

    Use this to understand the sequence and intent of changes
    before diving into the diff.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        offset: Number of commits to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum commits to return per call
            (defaults to ``tools.max_results`` config value).
    """
    repo = _repo(ctx)
    target = ctx.deps.state.review_target
    if target is None:
        return "No review target selected."

    try:
        entries = commit_log_between(repo, target.base_branch, target.head_ref)
    except KeyError as exc:
        return exc.args[0] if exc.args else str(exc)

    if not entries:
        return "No commits between base and head (branches are identical)."

    all_lines = [f"{e.sha[:8]} {e.author}: {e.message}" for e in entries]
    total = len(all_lines)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    page = all_lines[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} commits."
    result = "\n".join(page)
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


# ── File tools (require repo) ───────────────────────────────────────


def _validate_path(path: str) -> str | None:
    """Return an error message if *path* is invalid, else ``None``."""
    if ".." in PurePosixPath(path).parts:
        return f"Path '{path}' contains '..' — must be relative to repo root."
    return None


def _read_fs_file(path: str) -> tuple[list[str], str | None]:
    """Read a file from the local filesystem.

    Returns ``(lines, None)`` on success, or ``([], error_msg)``
    on failure.
    """
    p = Path(path)
    if not p.exists():
        return [], f"File '{path}' not found."
    try:
        data = p.read_bytes()
    except OSError as exc:
        return [], f"Cannot read '{path}': {exc}"
    if is_binary(data):
        return [], f"File '{path}' is binary — cannot display."
    return data.decode(errors="replace").splitlines(), None


def _list_fs_files(prefix: str, repo: pygit2.Repository | None = None) -> list[str]:
    """List files on the local filesystem matching *prefix*.

    *prefix* is treated as a directory path.  Returns sorted
    relative paths for all regular files under that directory.
    Respects ``.gitignore``, ``include``, and ``extend_exclude``
    via :func:`is_path_ignored`.
    """
    base = Path(prefix) if prefix else Path(".")
    if not base.is_dir():
        return []
    entries: list[str] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = str(PurePosixPath(p))
        if is_path_ignored(
            rel, repo, include=config.index.include, exclude=config.index.extend_exclude
        ):
            continue
        entries.append(rel)
    return entries


def _read_blob(repo: pygit2.Repository, ref: str, path: str) -> pygit2.Blob | str:
    """Return the blob for *path* at *ref*, or an error string."""
    blob = read_blob(repo, ref, path)
    if blob is None:
        return f"File '{path}' not found at ref '{ref}'."
    return blob


def _number_lines(lines: list[str], start: int) -> str:
    """Format *lines* with right-aligned line numbers starting at *start*."""
    end = start + len(lines)
    width = len(str(end))
    return "\n".join(f"{start + i:{width}d}│ {line}" for i, line in enumerate(lines))


def _format_file_page(path: str, all_lines: list[str], offset: int, max_lines: int) -> str:
    """Shared formatter for read_file — produces numbered output with pagination hint."""
    total = len(all_lines)
    selected = all_lines[offset : offset + max_lines]
    line_start = offset + 1  # 1-indexed display

    header = f"# {path}  (lines {line_start}-{line_start + len(selected) - 1} of {total})"
    body = _number_lines(selected, line_start)
    output = f"{header}\n{body}"
    shown_end = offset + len(selected)
    if shown_end < total:
        output += _limited(shown_end, total, hint=f"offset={shown_end} to continue")
    return output


@agent.tool(prepare=_require_repo)
def read_file(
    ctx: RunContext[AgentDeps],
    path: str,
    ref: str = "head",
    offset: int = 0,
    max_lines: int | None = None,
) -> str:
    """Read file content by path.

    Looks up the file in the git object store first.  If the
    path is not found in git, falls back to the local filesystem
    — this covers workspace files (e.g. ``.rbtr/REVIEW-*`` notes
    created with ``edit``) and any other untracked files.

    Works on any file: source code, config files, docs, lockfiles,
    and review notes.  Returns content with line numbers.  Output
    is capped at ``max_lines``.

    Use this to read any file by path.  For reading a specific
    symbol's source code by name, ``read_symbol`` is faster.
    To find where something is in a file first, use ``grep``
    or ``list_symbols``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_lines`` (up to the
    configured cap) to get more per page.

    Args:
        path: File path relative to repo root
            (e.g. ``src/main.py``, ``.rbtr/REVIEW-plan.md``).
            Must not contain ``..``.
        ref: Which version of the codebase to read — ``"head"``
            (default), ``"base"``, or a raw commit SHA.
        offset: Number of lines to skip (0-indexed, default 0).
            Use to paginate through large files.
        max_lines: Maximum number of lines to return
            (defaults to ``tools.max_lines`` config value).
    """
    if err := _validate_path(path):
        return err

    capped = (
        min(max_lines, config.tools.max_lines) if max_lines is not None else config.tools.max_lines
    )

    # Try git object store first.
    repo = _repo(ctx)
    resolved = _resolve_tool_ref(ctx, ref)
    blob_result = _read_blob(repo, resolved, path)
    if not isinstance(blob_result, str):
        data: bytes = blob_result.data
        if is_binary(data):
            return f"File '{path}' is binary — cannot display."
        return _format_file_page(path, data.decode(errors="replace").splitlines(), offset, capped)

    # Fall back to local filesystem.
    if is_path_ignored(
        path, repo, include=config.index.include, exclude=config.index.extend_exclude
    ):
        return blob_result  # treat ignored paths as not found
    fs_lines, fs_err = _read_fs_file(path)
    if fs_err:
        # Return the original git error — it's more informative.
        return blob_result
    return _format_file_page(path, fs_lines, offset, capped)


@agent.tool(prepare=_require_repo)
def grep(
    ctx: RunContext[AgentDeps],
    search: str | int | float,
    path: str = "",
    ref: str = "head",
    offset: int = 0,
    max_hits: int | None = None,
    context_lines: int | None = None,
) -> str:
    """Search for a substring in one file or across the repository.

    Performs a case-insensitive substring search.  Each match is
    shown with surrounding context lines (configurable).  When
    matches are close together, their context regions are merged
    to avoid duplicate lines.  Output includes line numbers.

    Results are capped at ``max_hits`` match groups.  Use
    ``offset`` to paginate through more.

    Three modes depending on ``path``:

    - **Exact file** (e.g. ``"src/api/handler.py"``): search
      within that single file.
    - **Directory prefix** (e.g. ``"src/"``): search all text
      files under that subtree.
    - **Empty string** (default): search all text files in the
      repository.

    Binary files are silently skipped.

    Files not found in the git tree are looked up on the local
    filesystem as a fallback — this covers workspace files (e.g.
    ``.rbtr/REVIEW-*`` notes) and other untracked files.

    For searching across the indexed codebase by keyword (with
    BM25 ranking), use ``search_codebase``.  For name-based
    symbol lookup, use ``search_symbols``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_hits`` (up to the
    configured cap) to get more per page.

    Args:
        search: Substring to find.  Case-insensitive — ``"config"``
            matches ``Config``, ``CONFIG``, ``config``.
        path: File path or directory prefix relative to repo root
            (e.g. ``src/api/handler.py``, ``.rbtr/``, ``""``).
            Empty string (default) searches all repo files.
            Must not contain ``..``.
        ref: Which version of the codebase to read — ``"head"``
            (default), ``"base"``, or a raw commit SHA.
        offset: Number of match groups to skip (default 0).
            Use to fetch the next page when a previous call was
            limited.
        max_hits: Maximum match groups to return per call
            (defaults to ``tools.max_grep_hits`` config value).
        context_lines: Number of lines to show above and below
            each match.  Defaults to the configured value.
    """
    search = str(search)  # coerce non-string args (e.g. model sends a number)
    if path and (err := _validate_path(path)):
        return err

    ctx_n = context_lines if context_lines is not None else config.tools.grep_context_lines
    needle = search.lower()
    capped_hits = (
        min(max_hits, config.tools.max_grep_hits)
        if max_hits is not None
        else config.tools.max_grep_hits
    )

    repo = _repo(ctx)
    resolved = _resolve_tool_ref(ctx, ref)

    if path:
        # Try as an exact file in git first.
        blob_result = _read_blob(repo, resolved, path)
        if not isinstance(blob_result, str):
            return _grep_blob(blob_result, path, needle, ctx_n, offset, capped_hits)

        # Not an exact git file — check whether any git files match the prefix.
        try:
            commit = resolve_commit(repo, resolved)
        except KeyError:
            return f"Ref '{ref}' not found."
        norm_prefix = path.rstrip("/")
        has_git_files = any(
            ep.startswith(norm_prefix + "/") or ep == norm_prefix
            for ep, _ in walk_tree(repo, commit.tree, "")
        )
        if has_git_files:
            return _grep_tree(repo, commit, path, needle, ctx_n, offset, capped_hits)

        # No git files under prefix — fall back to local filesystem.
        return _grep_filesystem(path, needle, ctx_n, offset, capped_hits, repo=repo)
    else:
        # Repo-wide search — git only.
        try:
            commit = resolve_commit(repo, resolved)
        except KeyError:
            return f"Ref '{ref}' not found."
        return _grep_tree(repo, commit, "", needle, ctx_n, offset, capped_hits)


def _grep_lines(
    all_lines: list[str], path: str, needle: str, ctx_n: int, offset: int, max_hits: int
) -> str:
    """Search *all_lines* for *needle*, returning formatted matches with context."""
    total = len(all_lines)
    match_indices = [i for i, line in enumerate(all_lines) if needle in line.lower()]
    if not match_indices:
        return f"No matches for '{needle}' in '{path}'."

    # Build merged context regions.
    regions: list[tuple[int, int]] = []
    for idx in match_indices:
        region_start = max(idx - ctx_n, 0)
        region_end = min(idx + ctx_n + 1, total)
        if regions and region_start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], region_end)
        else:
            regions.append((region_start, region_end))

    # Paginate by match group (region).
    total_groups = len(regions)
    page = regions[offset : offset + max_hits]
    if not page:
        return f"Offset {offset} exceeds {total_groups} match groups."

    n_matches = len(match_indices)
    header = f"# {path}  ({n_matches} match{'es' if n_matches != 1 else ''})"
    sections: list[str] = [header]
    for region_start, region_end in page:
        sections.append(_number_lines(all_lines[region_start:region_end], region_start + 1))

    result = "\n\n".join(sections)
    shown = offset + len(page)
    if shown < total_groups:
        result += _limited(shown, total_groups, hint=f"offset={shown} to see more match groups")
    return result


def _grep_blob(
    blob: pygit2.Blob, path: str, needle: str, ctx_n: int, offset: int, max_hits: int
) -> str:
    """Search a single blob for *needle*, returning formatted matches."""
    data: bytes = blob.data
    if is_binary(data):
        return f"File '{path}' is binary — cannot search."
    return _grep_lines(
        data.decode(errors="replace").splitlines(), path, needle, ctx_n, offset, max_hits
    )


def _grep_filesystem(
    path: str,
    needle: str,
    ctx_n: int,
    offset: int,
    max_hits: int,
    repo: pygit2.Repository | None = None,
) -> str:
    """Search local filesystem files under *path* for *needle*.

    Respects ``.gitignore``, ``include``, and ``extend_exclude``
    via :func:`is_path_ignored`.
    """
    # Check if path is an exact file first.
    p = Path(path)
    if p.is_file():
        file_lines, err = _read_fs_file(path)
        if err:
            return err
        return _grep_lines(file_lines, path, needle, ctx_n, offset, max_hits)

    # Otherwise treat as a directory prefix.
    files = _list_fs_files(path, repo=repo)
    if not files:
        return f"No matches for '{needle}' under '{path}'."

    # Directory — search all files under prefix.
    all_groups: list[tuple[str, list[str], list[tuple[int, int]], list[int]]] = []
    total_matches = 0
    for fpath in files:
        file_lines, err = _read_fs_file(fpath)
        if err:
            continue
        match_indices = [i for i, line in enumerate(file_lines) if needle in line.lower()]
        if not match_indices:
            continue
        total_matches += len(match_indices)
        total_file = len(file_lines)
        regions: list[tuple[int, int]] = []
        for idx in match_indices:
            rs = max(idx - ctx_n, 0)
            re_ = min(idx + ctx_n + 1, total_file)
            if regions and rs <= regions[-1][1]:
                regions[-1] = (regions[-1][0], re_)
            else:
                regions.append((rs, re_))
        all_groups.append((fpath, file_lines, regions, match_indices))

    if not all_groups:
        return f"No matches for '{needle}' under '{path}'."

    # Flatten and paginate — same as _grep_tree.
    flat: list[tuple[str, list[str], tuple[int, int]]] = []
    for fpath, file_lines, regions, _indices in all_groups:
        for region in regions:
            flat.append((fpath, file_lines, region))

    total_groups = len(flat)
    page = flat[offset : offset + max_hits]
    if not page:
        return f"Offset {offset} exceeds {total_groups} match groups."

    result_header = (
        f"Found {total_matches} match{'es' if total_matches != 1 else ''}"
        f" in {len(all_groups)} file{'s' if len(all_groups) != 1 else ''}"
    )
    file_sections: list[str] = []
    current_file: str | None = None
    current_parts: list[str] = []
    for fpath, file_lines, (rstart, rend) in page:
        if fpath != current_file:
            if current_parts:
                file_sections.append("\n\n".join(current_parts))
            current_parts = [f"# {fpath}"]
            current_file = fpath
        current_parts.append(_number_lines(file_lines[rstart:rend], rstart + 1))
    if current_parts:
        file_sections.append("\n\n".join(current_parts))

    body = "\n\n".join(file_sections)
    result = f"{result_header}\n\n{body}"
    shown = offset + len(page)
    if shown < total_groups:
        result += _limited(shown, total_groups, hint=f"offset={shown} to see more match groups")
    return result


def _grep_tree(
    repo: pygit2.Repository,
    commit: pygit2.Commit,
    prefix: str,
    needle: str,
    ctx_n: int,
    offset: int,
    max_hits: int,
) -> str:
    """Search all text files under *prefix* in the tree for *needle*."""
    max_file_size = config.index.max_file_size
    norm_prefix = prefix.rstrip("/")

    # Collect all match groups across all files.
    all_groups: list[tuple[str, list[str], list[tuple[int, int]], list[int]]] = []
    total_matches = 0

    for entry_path, blob in walk_tree(repo, commit.tree, ""):
        # Scope to prefix.
        if (
            norm_prefix
            and not entry_path.startswith(norm_prefix + "/")
            and entry_path != norm_prefix
        ):
            continue

        # Skip binary and oversized files.
        data: bytes = blob.data
        if len(data) > max_file_size or is_binary(data):
            continue

        file_lines = data.decode(errors="replace").splitlines()
        total_file = len(file_lines)
        match_indices = [i for i, line in enumerate(file_lines) if needle in line.lower()]
        if not match_indices:
            continue

        total_matches += len(match_indices)

        # Build merged context regions.
        regions: list[tuple[int, int]] = []
        for idx in match_indices:
            region_start = max(idx - ctx_n, 0)
            region_end = min(idx + ctx_n + 1, total_file)
            if regions and region_start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], region_end)
            else:
                regions.append((region_start, region_end))

        all_groups.append((entry_path, file_lines, regions, match_indices))

    if not all_groups:
        scope = f" under '{prefix}'" if prefix else ""
        return f"No matches for '{needle}'{scope}."

    # Flatten to (file, region) pairs for pagination.
    flat: list[tuple[str, list[str], tuple[int, int]]] = []
    for entry_path, file_lines, regions, _indices in all_groups:
        for region in regions:
            flat.append((entry_path, file_lines, region))

    total_groups = len(flat)
    page = flat[offset : offset + max_hits]
    if not page:
        return f"Offset {offset} exceeds {total_groups} match groups."

    result_header = (
        f"Found {total_matches} match{'es' if total_matches != 1 else ''}"
        f" in {len(all_groups)} file{'s' if len(all_groups) != 1 else ''}"
    )

    # Group page entries by file for readable output.
    file_sections: list[str] = []
    current_file: str | None = None
    current_parts: list[str] = []
    for entry_path, file_lines, (rstart, rend) in page:
        if entry_path != current_file:
            if current_parts:
                file_sections.append("\n\n".join(current_parts))
            current_parts = [f"# {entry_path}"]
            current_file = entry_path
        current_parts.append(_number_lines(file_lines[rstart:rend], rstart + 1))
    if current_parts:
        file_sections.append("\n\n".join(current_parts))

    body = "\n\n".join(file_sections)
    result = f"{result_header}\n\n{body}"
    shown = offset + len(page)
    if shown < total_groups:
        result += _limited(shown, total_groups, hint=f"offset={shown} to see more match groups")
    return result


@agent.tool(prepare=_require_repo)
def list_files(
    ctx: RunContext[AgentDeps],
    path: str = "",
    ref: str = "head",
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """List files in the repository or a subdirectory.

    Walks the git tree at the given ref and returns file paths,
    sorted alphabetically.  When ``path`` is set and no matching
    files are found in the git tree, falls back to the local
    filesystem — this covers workspace files (e.g.
    ``.rbtr/REVIEW-*`` notes) and other untracked files.

    Use this to explore project structure before reading specific
    files.  For a list of symbols in a specific file, use
    ``list_symbols``.

    **Pagination:** when the output ends with ``... limited
    (shown/total)``, pass the ``offset`` value from the hint to
    fetch the next page.  Increase ``max_results`` (up to the
    configured cap) to get more per page.

    Args:
        path: Directory prefix to scope the listing
            (e.g. ``src/api``, ``.rbtr/``).  Only files under
            this prefix are returned.  Empty string (default)
            lists from the repo root.
        ref: Which version of the codebase to read — ``"head"``
            (default), ``"base"``, or a raw commit SHA.
        offset: Number of entries to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum entries to return per call
            (defaults to ``tools.max_results`` config value).
    """
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )

    # Try git tree first.
    repo = _repo(ctx)
    resolved = _resolve_tool_ref(ctx, ref)
    try:
        commit = resolve_commit(repo, resolved)
    except KeyError:
        return f"Ref '{ref}' not found."

    prefix = path.rstrip("/")
    git_entries: list[str] = []
    for entry_path, _blob in walk_tree(repo, commit.tree, ""):
        if prefix and not entry_path.startswith(prefix + "/") and entry_path != prefix:
            continue
        git_entries.append(entry_path)

    if git_entries:
        git_entries.sort()
        return _format_file_list(git_entries, offset, limit)

    # Fall back to local filesystem when path is set.
    if path:
        fs_entries = _list_fs_files(path, repo=repo)
        if fs_entries:
            return _format_file_list(fs_entries, offset, limit)

    return f"No files found under '{path}'." if path else "No files in repository."


def _format_file_list(entries: list[str], offset: int, limit: int) -> str:
    """Format a paginated file listing."""
    total = len(entries)
    page = entries[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} files."
    header = f"Files ({total}):"
    listing = "\n".join(f"  {e}" for e in page)
    result = f"{header}\n{listing}"
    shown = offset + len(page)
    if shown < total:
        result += _limited(shown, total, hint=f"offset={shown} to continue")
    return result


# ── Review notes (always available) ──────────────────────────────────

_REVIEW_DIR = Path(".rbtr")


@agent.tool
def edit(
    ctx: RunContext[AgentDeps],
    path: str,
    new_text: str,
    old_text: str = "",
) -> str:
    """Edit or create a review notes file in the .rbtr workspace.

    Use this to maintain living review documents — plans,
    findings, checklists, draft comments — that persist across
    turns.  Files are plain text (Markdown recommended).

    Only files inside ``.rbtr/`` whose name starts with the
    configured workspace prefix (default ``REVIEW-``) are
    writable (e.g. ``.rbtr/REVIEW-plan.md``,
    ``.rbtr/REVIEW-findings.md``).

    Two modes:

    - **Create / append:** when ``old_text`` is empty, the file
      is created (or appended to if it already exists) with
      ``new_text``.
    - **Replace:** when ``old_text`` is non-empty, it must match
      exactly once in the file.  That occurrence is replaced with
      ``new_text``.

    Args:
        path: File path relative to the workspace root
            (e.g. ``.rbtr/REVIEW-plan.md``).  Must be inside
            ``.rbtr/`` and start with the workspace prefix.
        new_text: Content to write or insert.
        old_text: Exact text to find and replace.  Empty string
            (default) creates the file or appends to it.
    """
    prefix = config.tools.workspace_prefix
    # Validate path.
    p = PurePosixPath(path)
    parts = p.parts
    if len(parts) < 2 or parts[0] != ".rbtr":
        return f"Path must be inside .rbtr/ — got '{path}'."
    if not p.name.startswith(prefix):
        return f"Filename must start with '{prefix}' — got '{p.name}'."
    if ".." in parts:
        return f"Path '{path}' contains '..' — not allowed."

    resolved = _REVIEW_DIR / PurePosixPath(*parts[1:])

    if not old_text:
        # Create or append.
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if resolved.exists():
            existing = resolved.read_text()
            resolved.write_text(existing + new_text)
            return f"Appended to {path}."
        resolved.write_text(new_text)
        return f"Created {path}."

    # Replace exact match.
    if not resolved.exists():
        return f"File '{path}' does not exist — cannot replace."
    content = resolved.read_text()
    count = content.count(old_text)
    if count == 0:
        return f"old_text not found in '{path}'."
    if count > 1:
        return f"old_text matches {count} times in '{path}' — must be unique."
    resolved.write_text(content.replace(old_text, new_text, 1))
    return f"Replaced in {path}."
