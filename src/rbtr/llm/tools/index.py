"""Index tools — search, read_symbol, list_symbols, find_references, changed_symbols."""

from __future__ import annotations

from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.index.models import ChunkKind, EdgeKind
from rbtr.index.orchestrator import compute_diff
from rbtr.llm.agent import AgentDeps, agent
from rbtr.llm.tools.common import (
    get_store,
    head_commit,
    limited,
    require_index,
    resolve_tool_ref,
    validate_path,
)


@agent.tool(prepare=require_index)
def search(
    ctx: RunContext[AgentDeps],
    query: str,
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Search the codebase — finds symbols by name, keywords, or concepts.

    Fuses name matching, keyword search (BM25), and semantic
    similarity into a single ranked result list.  Works for exact
    identifiers (``IndexStore``), keyword queries (``retry timeout``),
    and natural-language concepts (``how does auth work``).

    Args:
        query: What to search for — an identifier name, keywords,
            or a natural-language description.
        offset: Number of results to skip (default 0).  Use to
            fetch the next page when a previous call was limited.
        max_results: Maximum results to return per call
            (defaults to `tools.max_results` config value).
    """
    store = get_store(ctx)
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )
    results = store.search(head_commit(ctx), query, top_k=offset + limit + 1)
    if not results:
        return f"No results for '{query}'."
    total = len(results)
    page = results[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} results."
    lines: list[str] = []
    for r in page:
        c = r.chunk
        scope = f"{c.scope}." if c.scope else ""
        lines.append(f"[{r.score:.3f}] {c.kind} {scope}{c.name}  ({c.file_path}:{c.line_start})")
    result = "\n".join(lines)
    shown = offset + len(page)
    if shown < total:
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result


@agent.tool(prepare=require_index)
def read_symbol(
    ctx: RunContext[AgentDeps],
    name: str,
    ref: str = "head",
) -> str:
    """Read the full source code of a symbol from the index.

    Args:
        name: Short name to match (e.g. `MQ`,
            `parse_config`).  Case-insensitive substring match.
            Returns the first matching code symbol.  Use short
            names, not fully-qualified paths — `MQ` not
            `lib.mq.MQ`.
        ref: Which version of the codebase to read
            (defaults to `"head"`).
    """
    store = get_store(ctx)
    resolved = resolve_tool_ref(ctx, ref)
    matches = store.search_by_name(resolved, name)

    code_kinds = frozenset({ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD})
    symbols = [c for c in matches if c.kind in code_kinds]

    if not symbols:
        return f"No symbol matching '{name}'. Use search with a shorter substring."

    sym = symbols[0]
    header = f"# {sym.kind} {sym.scope + '.' if sym.scope else ''}{sym.name}"
    location = f"# {sym.file_path}:{sym.line_start}-{sym.line_end}"
    content_lines = sym.content.splitlines()
    max_lines = config.tools.max_lines
    if len(content_lines) > max_lines:
        body = "\n".join(content_lines[:max_lines])
        total = len(content_lines)
        return f"{header}\n{location}\n\n{body}" + limited(
            max_lines,
            total,
            hint=f"use read_file('{sym.file_path}', offset={max_lines}) to continue",
        )
    return f"{header}\n{location}\n\n{sym.content}"


@agent.tool(prepare=require_index)
def list_symbols(
    ctx: RunContext[AgentDeps],
    path: str,
    ref: str = "head",
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """List the symbols (functions, classes, methods) in a file.

    Args:
        path: File path relative to repo root
            (e.g. `src/api/handler.py`).
        ref: Which version of the codebase to read
            (defaults to `"head"`).
        offset: Number of symbols to skip (default 0).
        max_results: Maximum symbols to return per call
            (defaults to `tools.max_results` config value).
    """
    if err := validate_path(path):
        return err

    store = get_store(ctx)
    resolved = resolve_tool_ref(ctx, ref)
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
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result


@agent.tool(prepare=require_index)
def find_references(
    ctx: RunContext[AgentDeps],
    name: str,
    kind: EdgeKind | None = None,
    ref: str = "head",
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """Find symbols that reference a given symbol via the dependency graph.

    Args:
        name: Short symbol name to look up
            (e.g. `Config`, `parse_request`).
        kind: Edge type to filter by.  `None` (default)
            returns all edge types.
        ref: Which version of the codebase to query
            (defaults to `"head"`).
        offset: Number of results to skip (default 0).
        max_results: Maximum results to return per call
            (defaults to `tools.max_results` config value).
    """
    store = get_store(ctx)
    resolved = resolve_tool_ref(ctx, ref)

    targets = store.search_by_name(resolved, name)
    if not targets:
        return f"Symbol '{name}' not found."

    target_ids = {c.id for c in targets}

    edges = store.get_edges(resolved)
    matching = [e for e in edges if e.target_id in target_ids and (kind is None or e.kind == kind)]

    if not matching:
        if kind:
            return f"No '{kind.value}' references found for '{name}'."
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
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result


@agent.tool(prepare=require_index)
def changed_symbols(
    ctx: RunContext[AgentDeps],
    offset: int = 0,
    max_lines: int | None = None,
) -> str:
    """List symbols changed between base and head.

    Args:
        offset: Number of output lines to skip (default 0).
        max_lines: Maximum lines to return per call
            (defaults to `tools.max_lines` config value).
    """
    store = get_store(ctx)
    target = ctx.deps.state.review_target
    if target is None:
        return "No review target selected."

    sd = compute_diff(target.base_commit, target.head_commit, store)
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
        result += limited(
            offset + len(page), total, hint=f"offset={offset + len(page)} to continue"
        )
    return result
