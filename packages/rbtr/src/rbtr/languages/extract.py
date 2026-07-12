"""Per-file extraction: pick a language's strategy and run it.

`extract_file` is the per-file entry point the indexer calls for every file
(and that plugin tests and third-party language packages drive directly). It
selects the primary strategy — a registered chunker, the tree-sitter query,
or plaintext — via `extract_primary` / `extract_query`, then layers
embedded-language injections (`extract_injections`) on top. The build loop in
`rbtr.index.orchestrator` consumes `extract_file`; this module is the
extraction runtime, kept beside the language declarations it runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Parser, QueryCursor

from rbtr.git import FileEntry
from rbtr.index.models import Chunk
from rbtr.languages.chunks import chunk_plaintext, host_presence_chunk
from rbtr.languages.manager import LanguageManager, get_manager
from rbtr.languages.registration import ChunkExtraction, QueryExtraction
from rbtr.languages.treesitter import _get_query, extract_symbols

if TYPE_CHECKING:
    from tree_sitter import Node, Range


def extract_query(
    language: str,
    file_path: str,
    blob_sha: str,
    content: bytes,
    ranges: list[Range] | None = None,
    doc_comment_node_types: frozenset[str] | None = None,
) -> Iterator[Chunk]:
    """Run *language*'s tree-sitter query over *content*.

    The single query-path extraction entry. `ranges=None` parses the
    whole file; a range list restricts parsing to those spans (an
    embedded block, e.g. an SFC `<script>`), reporting absolute
    positions. Yields nothing if *language* has no grammar or query.
    The injection runner (`extract_injections`) calls this to delegate
    each embedded block to its language.

    *doc_comment_node_types* defaults to the registration's; pass an
    explicit (possibly empty) set to override leading-comment attachment.
    """
    mgr = get_manager()
    reg = mgr.get_registration(language)
    grammar = mgr.load_grammar(language)
    if reg is None or not isinstance(reg.extraction, QueryExtraction) or grammar is None:
        return
    yield from extract_symbols(
        reg,
        file_path,
        blob_sha,
        content,
        grammar,
        doc_comment_node_types=doc_comment_node_types,
        included_ranges=ranges,
    )


def extract_primary(
    language: str,
    file_path: str,
    blob_sha: str,
    content: bytes,
    ranges: list[Range] | None = None,
) -> list[Chunk] | None:
    """Run *language*'s primary extraction (its chunker or query) over *content*.

    Restricted to *ranges* when given — an embedded block — so it doubles as
    the injection delegate: the full target plugin runs on the range. Every
    returned chunk carries *language* (query extraction sets it already; a
    chunker's blank chunks are filled here), so a delegated chunker target is
    labelled correctly. Returns None when the language has neither a chunker
    nor a query, leaving the caller to fall back to plaintext.
    """
    mgr = get_manager()
    grammar = mgr.load_grammar(language)
    reg = mgr.get_registration(language)
    extraction = reg.extraction if reg is not None else None
    if isinstance(extraction, ChunkExtraction) and grammar is not None:
        chunks = list(
            extraction.chunker(
                file_path, blob_sha, content.decode(errors="replace"), grammar, ranges
            )
        )
    elif isinstance(extraction, QueryExtraction) and grammar is not None:
        chunks = list(extract_query(language, file_path, blob_sha, content, ranges=ranges))
    else:
        return None
    for chunk in chunks:
        if not chunk.language:
            chunk.language = language
    return chunks


def _resolve_injection_hint(mgr: LanguageManager, captures: dict[str, list[Node]]) -> str | None:
    """Resolve a dynamic `@injection.language` hint to a language id.

    A markdown fence tags its block with a free-form name (` ```python `,
    ` ```py `). Treat the name as a language id, then as a file extension, so
    both spellings reach the python plugin. An unknown hint returns None and
    the block is left unparsed.
    """
    nodes = captures.get("injection.language")
    if not nodes or nodes[0].text is None:
        return None
    hint = nodes[0].text.decode().strip().lower()
    return hint if mgr.get_registration(hint) else mgr.detect_language(f"x.{hint}")


_MAX_INJECTION_DEPTH = 5
"""Recursion cap for nested injection (a host embedded in a host). Insurance
only — delegation terminates naturally as ranges shrink and a target with no
`injection_query` stops."""


def extract_injections(
    language: str,
    file_path: str,
    blob_sha: str,
    content: bytes,
    ranges: list[Range] | None = None,
    _depth: int = 0,
) -> Iterator[Chunk]:
    """Yield chunks for code embedded in *content* via *language*'s injections.

    A host language declares an `injection_query` marking each embedded block
    and the language to parse it as; each block's range is delegated to that
    language's *full* primary extraction (`extract_primary` — chunker or
    query), so a `<script lang="ts">` block yields TypeScript chunks and a
    fenced ` ```html ` block yields HTML chunks, both at real line numbers.
    The target is a static `#set! injection.language`, or — for a free-form
    hint like a markdown fence — a captured `@injection.language` resolved to
    a language id. A block matching several rules uses the highest priority.

    Delegation recurses: the target's own injection runs on the block too, so
    an HTML block containing an inline `<script>` also yields its js, bounded
    by `_MAX_INJECTION_DEPTH`. *ranges* restricts the host parse to a block
    (used by that recursion); None parses the whole file.
    """
    if _depth > _MAX_INJECTION_DEPTH:
        return
    mgr = get_manager()
    reg = mgr.get_registration(language)
    if reg is None or reg.injection_query is None:
        return
    grammar = mgr.load_grammar(language)
    if grammar is None:
        return

    query = _get_query(grammar, reg.injection_query)
    parser = Parser(grammar)
    if ranges is not None:
        parser.included_ranges = ranges
    tree = parser.parse(content)

    winner: dict[tuple[int, int], tuple[int, str, Range]] = {}
    for pattern, captures in QueryCursor(query).matches(tree.root_node):
        settings = query.pattern_settings(pattern)
        target = settings.get("injection.language") or _resolve_injection_hint(mgr, captures)
        if target is None:
            continue
        priority = int(settings.get("injection.priority") or "0")
        for block in captures.get("injection.content", []):
            if block.text is None or not block.text.strip():
                continue
            span = (block.start_byte, block.end_byte)
            if span not in winner or priority > winner[span][0]:
                winner[span] = (priority, target, block.range)

    for _priority, target, block_range in winner.values():
        delegated = extract_primary(target, file_path, blob_sha, content, ranges=[block_range])
        if delegated is not None:
            yield from delegated
        yield from extract_injections(
            target, file_path, blob_sha, content, ranges=[block_range], _depth=_depth + 1
        )


def extract_file(entry: FileEntry, language: str) -> list[Chunk]:
    """Extract a file's chunks, always including one in its own language.

    The per-file extraction entry point: the indexer calls this for every
    file, and it is public so plugin tests (and third-party language
    packages) drive the *real* pipeline rather than a copy. Given a file's
    content and language it picks the primary strategy — a registered
    chunker, the tree-sitter query, or plaintext line chunks — then adds
    any embedded-language injections (an SFC's `<script>`/`<style>`) on
    top. If none of that produced a chunk in the file's own *language*, a
    content-less host-presence chunk is appended so the dedup gate can skip
    the file on later builds instead of re-parsing it. The caller handles
    blob dedup and deletes stale chunks first.
    """
    text = entry.content.decode(errors="replace")
    reg = get_manager().get_registration(language)
    has_injection = reg is not None and reg.injection_query is not None

    primary = extract_primary(language, entry.path, entry.blob_sha, entry.content)
    if primary is not None:
        chunks = primary
    elif has_injection:
        chunks = []
    else:
        chunks = list(chunk_plaintext(entry.path, entry.blob_sha, text))

    if has_injection:
        chunks += extract_injections(language, entry.path, entry.blob_sha, entry.content)

    if not any(chunk.language == language for chunk in chunks):
        chunks.append(host_presence_chunk(entry.path, entry.blob_sha, language))

    return chunks
