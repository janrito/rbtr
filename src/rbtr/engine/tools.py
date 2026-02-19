"""LLM tools — expose the code index and git history to the agent.

Each tool is registered on the shared ``agent`` instance via
``@agent.tool``.  Tools receive ``RunContext[AgentDeps]`` and
read ``session.index`` / ``session.review_target`` / ``session.repo``.

Index tools are hidden when no index is loaded.  Git tools are
hidden when no repo is available.
"""

from __future__ import annotations

import pygit2
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from pygit2.enums import SortMode

from rbtr.config import config
from rbtr.index.models import ChunkKind, EdgeKind
from rbtr.index.store import IndexStore

from .agent import AgentDeps, agent

# ── Prepare functions ────────────────────────────────────────────────


async def _require_index(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when an index is available."""
    if ctx.deps.session.index is None or ctx.deps.session.review_target is None:
        return None
    return tool_def


async def _require_repo(
    ctx: RunContext[AgentDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Return the tool definition only when a repo + review target is available."""
    if ctx.deps.session.repo is None or ctx.deps.session.review_target is None:
        return None
    return tool_def


# ── Accessor helpers ─────────────────────────────────────────────────


def _head_ref(ctx: RunContext[AgentDeps]) -> str:
    """Return the head branch ref from the review target."""
    target = ctx.deps.session.review_target
    if target is None:  # pragma: no cover — guarded by prepare
        msg = "no review target"
        raise RuntimeError(msg)
    return target.head_branch


def _store(ctx: RunContext[AgentDeps]) -> IndexStore:
    """Return the index store."""
    store = ctx.deps.session.index
    if store is None:  # pragma: no cover — guarded by _require_index
        msg = "no index store"
        raise RuntimeError(msg)
    return store


def _repo(ctx: RunContext[AgentDeps]) -> pygit2.Repository:
    """Return the git repo."""
    repo = ctx.deps.session.repo
    if repo is None:  # pragma: no cover — guarded by _require_repo
        msg = "no repository"
        raise RuntimeError(msg)
    return repo


# ── Tools ────────────────────────────────────────────────────────────


@agent.tool(prepare=_require_index)
def search_symbols(
    ctx: RunContext[AgentDeps],
    name: str,
) -> str:
    """Search for functions, classes, and methods by name substring.

    Use short, simple names — NOT fully-qualified module paths.
    For example, use ``MQ`` not ``lib.mq.MQ``, or ``crawl`` not
    ``crawler.module.crawl``.  The search is case-insensitive and
    matches anywhere in the symbol name.

    Args:
        name: Short substring to match (e.g. ``parse``, ``Client``, ``MQ``).
    """
    store = _store(ctx)
    chunks = store.search_by_name(_head_ref(ctx), name)
    if not chunks:
        return f"No symbols matching '{name}'. Try a shorter or different substring."
    lines: list[str] = []
    for c in chunks[:20]:
        scope = f"{c.scope}." if c.scope else ""
        lines.append(f"{c.kind} {scope}{c.name}  ({c.file_path}:{c.line_start})")
    return "\n".join(lines)


@agent.tool(prepare=_require_index)
def search_codebase(
    ctx: RunContext[AgentDeps],
    query: str,
) -> str:
    """Full-text keyword search across the indexed codebase (BM25).

    Args:
        query: Keywords to search for in symbol names and content.
    """
    store = _store(ctx)
    results = store.search_fulltext(_head_ref(ctx), query, top_k=10)
    if not results:
        return f"No results for '{query}'."
    lines: list[str] = []
    for chunk, score in results:
        scope = f"{chunk.scope}." if chunk.scope else ""
        lines.append(
            f"[{score:.2f}] {chunk.kind} {scope}{chunk.name}"
            f"  ({chunk.file_path}:{chunk.line_start})"
        )
    return "\n".join(lines)


@agent.tool(prepare=_require_index)
def search_similar(
    ctx: RunContext[AgentDeps],
    query: str,
    top_k: int = 10,
) -> str:
    """Semantic similarity search — find code conceptually related to a natural-language query.

    Uses embedding vectors to find symbols whose meaning is close to the
    query, even if they don't share keywords.  Complements ``search_codebase``
    (keyword/BM25) for when you know *what* you're looking for but not the
    exact names.

    Args:
        query: Natural-language description of what you're looking for.
        top_k: Maximum number of results (default 10).
    """
    store = _store(ctx)
    try:
        results = store.search_by_text(_head_ref(ctx), query, top_k=top_k)
    except Exception:
        return (
            "Semantic search unavailable (embedding model not loaded). "
            "Use search_codebase (keyword search) or search_symbols instead."
        )
    if not results:
        return f"No similar symbols for '{query}'."
    lines: list[str] = []
    for chunk, score in results:
        scope = f"{chunk.scope}." if chunk.scope else ""
        lines.append(
            f"[{score:.3f}] {chunk.kind} {scope}{chunk.name}"
            f"  ({chunk.file_path}:{chunk.line_start})"
        )
    return "\n".join(lines)


@agent.tool(prepare=_require_index)
def get_dependents(
    ctx: RunContext[AgentDeps],
    symbol_name: str,
) -> str:
    """Find what imports or depends on a given symbol.

    Use a short name, not a module path.

    Args:
        symbol_name: Short symbol name (e.g. ``MQ``, ``Config``).
    """
    store = _store(ctx)
    ref = _head_ref(ctx)

    targets = store.search_by_name(ref, symbol_name)
    if not targets:
        return f"Symbol '{symbol_name}' not found."

    target_ids = {c.id for c in targets}
    edges = store.get_edges(ref)
    import_edges = [e for e in edges if e.kind == EdgeKind.IMPORTS and e.target_id in target_ids]

    if not import_edges:
        return f"No dependents found for '{symbol_name}'."

    all_chunks = {c.id: c for c in store.get_chunks(ref)}
    lines: list[str] = []
    for edge in import_edges:
        src = all_chunks.get(edge.source_id)
        if src:
            lines.append(f"{src.kind} {src.name}  ({src.file_path}:{src.line_start})")
    return "\n".join(lines) or f"No dependents found for '{symbol_name}'."


@agent.tool(prepare=_require_index)
def get_callers(
    ctx: RunContext[AgentDeps],
    symbol_name: str,
) -> str:
    """Find test files and doc sections that reference a symbol.

    Use a short name, not a module path.

    Args:
        symbol_name: Short symbol name (e.g. ``MQ``, ``Config``).
    """
    store = _store(ctx)
    ref = _head_ref(ctx)

    targets = store.search_by_name(ref, symbol_name)
    if not targets:
        return f"Symbol '{symbol_name}' not found."

    target_ids = {c.id for c in targets}
    edges = store.get_edges(ref)
    ref_edges = [
        e
        for e in edges
        if e.kind in (EdgeKind.TESTS, EdgeKind.DOCUMENTS) and e.target_id in target_ids
    ]

    if not ref_edges:
        return f"No tests or docs reference '{symbol_name}'."

    all_chunks = {c.id: c for c in store.get_chunks(ref)}
    lines: list[str] = []
    for edge in ref_edges:
        src = all_chunks.get(edge.source_id)
        if src:
            label = "tests" if edge.kind == EdgeKind.TESTS else "documents"
            lines.append(f"[{label}] {src.name}  ({src.file_path}:{src.line_start})")
    return "\n".join(lines) or f"No tests or docs reference '{symbol_name}'."


@agent.tool(prepare=_require_index)
def get_blast_radius(
    ctx: RunContext[AgentDeps],
    file_path: str,
) -> str:
    """Show what depends on symbols in a file — imports, tests, and docs.

    Args:
        file_path: Path of the file to check (e.g. 'src/rbtr/index/store.py').
    """
    store = _store(ctx)
    ref = _head_ref(ctx)

    file_chunks = store.get_chunks(ref, file_path=file_path)
    if not file_chunks:
        return f"No indexed symbols in '{file_path}'."

    file_chunk_ids = {c.id for c in file_chunks}
    edges = store.get_edges(ref)
    inbound = [e for e in edges if e.target_id in file_chunk_ids]

    if not inbound:
        return f"Nothing depends on symbols in '{file_path}'."

    all_chunks = {c.id: c for c in store.get_chunks(ref)}
    lines: list[str] = []
    for edge in inbound:
        src = all_chunks.get(edge.source_id)
        tgt = all_chunks.get(edge.target_id)
        if src and tgt:
            lines.append(f"[{edge.kind}] {src.name} ({src.file_path}) → {tgt.name}")
    return "\n".join(lines) or f"Nothing depends on symbols in '{file_path}'."


@agent.tool(prepare=_require_index)
def read_symbol(
    ctx: RunContext[AgentDeps],
    symbol_name: str,
) -> str:
    """Read the full source code of a symbol (function, class, or method).

    Use a short name — NOT a fully-qualified path.  For example,
    ``MQ`` not ``lib.mq.MQ``, or ``handle_request`` not
    ``server.handlers.handle_request``.  If unsure of the exact
    name, use ``search_symbols`` first.

    Args:
        symbol_name: Short name to match (e.g. ``MQ``, ``parse_config``).
    """
    store = _store(ctx)
    matches = store.search_by_name(_head_ref(ctx), symbol_name)

    code_kinds = frozenset({ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD})
    symbols = [c for c in matches if c.kind in code_kinds]

    if not symbols:
        return (
            f"No symbol matching '{symbol_name}'. "
            f"Use search_symbols with a shorter substring, or "
            f"list_indexed_files to see what files are indexed."
        )

    sym = symbols[0]
    header = f"# {sym.kind} {sym.scope + '.' if sym.scope else ''}{sym.name}"
    location = f"# {sym.file_path}:{sym.line_start}-{sym.line_end}"
    return f"{header}\n{location}\n\n{sym.content}"


@agent.tool(prepare=_require_index)
def list_indexed_files(
    ctx: RunContext[AgentDeps],
    path_filter: str = "",
) -> str:
    """List files in the index, optionally filtered by path substring.

    Use this to discover what files are available before searching
    for symbols.  Shows each file with its symbol count.

    Args:
        path_filter: Optional substring to filter paths (e.g. ``mq``, ``lib/``).
    """
    store = _store(ctx)
    chunks = store.get_chunks(_head_ref(ctx))
    if not chunks:
        return "No files indexed yet."

    # Group by file path.
    files: dict[str, int] = {}
    for c in chunks:
        files[c.file_path] = files.get(c.file_path, 0) + 1

    if path_filter:
        files = {p: n for p, n in files.items() if path_filter.lower() in p.lower()}
        if not files:
            return f"No indexed files matching '{path_filter}'."

    lines = [f"  {path}  ({count} symbols)" for path, count in sorted(files.items())]
    total = len(lines)
    if total > 50:
        lines = lines[:50]
        lines.append(f"  ... and {total - 50} more files")
    return f"Indexed files ({total}):\n" + "\n".join(lines)


@agent.tool(prepare=_require_index)
def detect_language(
    ctx: RunContext[AgentDeps],
    file_path: str,
) -> str:
    """Detect the programming language of a file.

    Args:
        file_path: Path to check (e.g. 'src/main.rs').
    """
    from rbtr.plugins.manager import get_manager  # deferred: avoids circular at import time

    mgr = get_manager()
    lang_id = mgr.detect_language(file_path)
    if lang_id is None:
        return f"Unknown language for '{file_path}'."
    return lang_id


@agent.tool(prepare=_require_index)
def semantic_diff(
    ctx: RunContext[AgentDeps],
) -> str:
    """Structural diff between the review target's base and head branches.

    Returns:
    - Added / removed / modified symbols (functions, classes, methods).
    - Stale docs: documentation that references modified code but wasn't updated.
    - Missing tests: new functions/methods with no test coverage.
    - Broken edges: imports in head that point at removed symbols.
    """
    from rbtr.index.orchestrator import compute_diff  # deferred: avoid circular

    store = _store(ctx)
    target = ctx.deps.session.review_target
    if target is None:
        return "No review target selected."

    sd = compute_diff(target.base_branch, target.head_branch, store)
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

    return "\n\n".join(sections)


# ── Git tools ────────────────────────────────────────────────────────


@agent.tool(prepare=_require_repo)
def diff(
    ctx: RunContext[AgentDeps],
    ref: str = "",
) -> str:
    """Show a unified diff.

    With no argument, diffs base branch against head branch (the
    review target).  With a single ref, shows that commit's diff
    against its parent.  With ``base..head`` syntax, diffs between
    the two refs.

    Args:
        ref: A commit SHA, branch name, or ``base..head`` range.
             Empty string diffs the review target's base vs head.
    """
    repo = _repo(ctx)
    target = ctx.deps.session.review_target

    if not ref:
        # Default: base → head of review target.
        if target is None:
            return "No review target selected."
        base_ref = target.base_branch
        head_ref = target.head_branch
    elif ".." in ref:
        parts = ref.split("..", 1)
        base_ref = parts[0]
        head_ref = parts[1]
    else:
        # Single ref — diff against parent.
        try:
            commit = repo.revparse_single(ref).peel(pygit2.Commit)
        except (KeyError, ValueError, pygit2.GitError):
            return f"Ref '{ref}' not found."
        if not commit.parent_ids:
            return f"Commit {ref} has no parent (initial commit)."
        parent_obj = repo.get(commit.parent_ids[0])
        if parent_obj is None:
            return f"Parent of {ref} not found."
        parent = parent_obj.peel(pygit2.Commit)
        return _format_diff(repo.diff(parent, commit))

    try:
        base_commit = repo.revparse_single(base_ref).peel(pygit2.Commit)
        head_commit = repo.revparse_single(head_ref).peel(pygit2.Commit)
    except (KeyError, ValueError, pygit2.GitError) as exc:
        return f"Could not resolve refs: {exc}"

    return _format_diff(repo.diff(base_commit, head_commit))


def _format_diff(d: pygit2.Diff) -> str:
    """Format a pygit2 Diff as a truncated unified diff string."""
    stats = d.stats
    header = f"{stats.files_changed} files changed, +{stats.insertions} -{stats.deletions}"
    patch = d.patch or "(empty diff)"
    max_chars = config.tools.max_diff_chars
    if len(patch) > max_chars:
        patch = patch[:max_chars] + f"\n\n... truncated ({len(patch)} chars total)"
    return f"{header}\n\n{patch}"


@agent.tool(prepare=_require_repo)
def commit_log(
    ctx: RunContext[AgentDeps],
) -> str:
    """Show the commit log between the review target's base and head branches.

    Returns a reverse-chronological list of commits on the head
    branch that are not on the base branch.
    """
    repo = _repo(ctx)
    target = ctx.deps.session.review_target
    if target is None:
        return "No review target selected."

    try:
        head_commit = repo.revparse_single(target.head_branch).peel(pygit2.Commit)
        base_commit = repo.revparse_single(target.base_branch).peel(pygit2.Commit)
    except (KeyError, ValueError, pygit2.GitError) as exc:
        return f"Could not resolve refs: {exc}"

    # Find merge base to know where to stop.
    merge_base = repo.merge_base(head_commit.id, base_commit.id)
    stop_at = merge_base if merge_base else None

    lines: list[str] = []
    for commit in repo.walk(head_commit.id, SortMode.TOPOLOGICAL):
        if stop_at and commit.id == stop_at:
            break
        msg = commit.message.strip().split("\n", 1)[0]
        sha = str(commit.id)[:8]
        author = commit.author.name
        lines.append(f"{sha} {author}: {msg}")
        max_commits = config.tools.max_log_commits
        if len(lines) >= max_commits:
            lines.append(f"... truncated at {max_commits} commits")
            break

    if not lines:
        return "No commits between base and head (branches are identical)."
    return "\n".join(lines)
