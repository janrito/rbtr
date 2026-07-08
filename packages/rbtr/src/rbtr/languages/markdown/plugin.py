"""Markdown language plugin.

Splits Markdown by heading hierarchy using tree-sitter.  Headed
sections produce one `doc_section` chunk per heading, containing
the heading and its direct content (excluding nested subsections).
Headingless documents fall back to `chunk_plaintext`.

Extracted chunks::

    # Title                         → doc_section "Title", scope ""
    Intro text.                       (content: heading + intro)

    ## Section A                    → doc_section "Section A", scope "Title"
    Body A.                           (content: heading + body, excludes
                                       child sections)

    First paragraph.                → raw_chunk (plaintext fallback)
    Second paragraph.               → raw_chunk
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Language, Parser, Query, QueryCursor

from rbtr.index.chunks import chunk_plaintext
from rbtr.index.identity import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind, ImportMeta
from rbtr.languages.queries import load_query
from rbtr.languages.registration import LanguageRegistration

if TYPE_CHECKING:
    from tree_sitter import Node, Range

# URL schemes that indicate external links (not cross-references).
_EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "ftp://")

# Block-level query: captures headed sections.
_SECTION_QUERY = load_query(__package__, "sections")

# Injection query: a fenced code block delegates its content to the language
# named in the fence's info string (```python, ```sh). The info-string name
# is captured dynamically and resolved to a language id by the runner.


# ── Chunking ─────────────────────────────────────────────────────────


def _has_headings(root: Node) -> bool:
    """Return True if any section in the tree has an atx_heading."""
    for child in root.children:
        if child.type == "section":
            if any(gc.type == "atx_heading" for gc in child.children):
                return True
            if _has_headings(child):
                return True
    return False


def _section_depth(node: Node) -> int:
    """Count ancestor nodes of the same type.

    In the tree-sitter markdown grammar, sections nest:
    a `## Sub` section is a child `section` node of the
    `# Top` section.  The depth (number of same-typed
    ancestors) maps directly to the heading level minus
    one: depth 0 = `#`, depth 1 = `##`, etc.
    """
    depth = 0
    ancestor = node.parent
    while ancestor is not None:
        if ancestor.type == node.type:
            depth += 1
        ancestor = ancestor.parent
    return depth


def _section_own_content(node: Node, content_bytes: bytes) -> str:
    """Return the section's own text, excluding nested subsections.

    A parent section's byte span covers its children.  Trimming
    at the first child of the same type gives us only the
    heading and the prose that belongs directly to this section.
    """
    end_byte = node.end_byte
    for child in node.children:
        if child.type == node.type:
            end_byte = child.start_byte
            break
    return content_bytes[node.start_byte : end_byte].decode("utf-8", errors="replace").strip()


def _extract_sections(
    content_bytes: bytes,
    grammar: Language,
    file_path: str,
    blob_sha: str,
    ranges: list[Range] | None,
) -> Iterator[Chunk]:
    """Run the section query and yield one chunk per headed section.

    The tree-sitter markdown grammar represents headings as
    nested `section` nodes.  This function queries for all
    sections that contain an `atx_heading` and produces one
    `doc_section` chunk per match.

    **Scope** is the parent heading chain, built by maintaining
    a stack of heading names indexed by depth.  When a section
    at depth *d* is encountered, the stack is truncated to *d*
    entries (popping deeper headings from a previous branch),
    the current scope is read from the remaining stack, and
    the new heading is pushed.

    **Content** is trimmed to exclude nested subsections — see
    `_section_own_content`.
    """
    query = Query(grammar, _SECTION_QUERY)
    parser = Parser(grammar)
    if ranges is not None:
        parser.included_ranges = ranges
    tree = parser.parse(content_bytes)
    matches = QueryCursor(query).matches(tree.root_node)

    # Heading names indexed by depth.  Tracks the current
    # branch of the heading tree so we can reconstruct the
    # scope ("Top::Mid") for each section.
    scope_stack: list[str] = []

    for _pattern_idx, capture_dict in matches:
        section_nodes = capture_dict.get("doc_section", [])
        name_nodes = capture_dict.get("_section_name", [])
        if not section_nodes or not name_nodes:
            continue

        node = section_nodes[0]
        name = name_nodes[0].text.decode("utf-8", errors="replace") if name_nodes[0].text else ""

        # Maintain the scope stack: pop back to this section's
        # depth, read scope, then push the current heading.
        depth = _section_depth(node)
        while len(scope_stack) > depth:
            scope_stack.pop()
        scope_segments = list(scope_stack)
        scope_stack.append(name)

        text = _section_own_content(node, content_bytes)
        if not text:
            continue

        line_start = node.start_point[0] + 1
        yield Chunk.model_validate(
            {
                "blob_sha": blob_sha,
                "file_path": file_path,
                "kind": ChunkKind.DOC_SECTION,
                "name": name,
                "scope": scope_segments,
                "language": "markdown",
                "content": text,
                "line_start": line_start,
                "line_end": node.end_point[0] + 1,
            }
        )


def chunk_markdown(
    file_path: str,
    blob_sha: str,
    content: str,
    grammar: Language,
    ranges: list[Range] | None = None,
) -> Iterator[Chunk]:
    """Split Markdown by heading hierarchy using tree-sitter."""
    if not content.strip():
        return

    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    if ranges is not None:
        parser.included_ranges = ranges
    tree = parser.parse(content_bytes)

    if _has_headings(tree.root_node):
        yield from _extract_sections(content_bytes, grammar, file_path, blob_sha, ranges)
    else:
        yield from chunk_plaintext(file_path, blob_sha, content)

    # Extract local links as IMPORT chunks using the inline parser.
    yield from _extract_links(content_bytes, file_path, blob_sha, ranges)


# ── Link extraction ──────────────────────────────────────────────────

_LINK_QUERY = load_query(__package__, "links")


def _extract_links(
    content_bytes: bytes,
    file_path: str,
    blob_sha: str,
    ranges: list[Range] | None,
) -> Iterator[Chunk]:
    """Extract local links as IMPORT chunks using the inline parser.

    Parses the full content with `tree_sitter_markdown.inline_language()`
    and queries for `inline_link` nodes.  External URLs and same-file
    fragment-only links are skipped.
    """
    import tree_sitter_markdown  # deferred: heavy native lib

    inline_lang = Language(tree_sitter_markdown.inline_language())
    inline_parser = Parser(inline_lang)
    if ranges is not None:
        inline_parser.included_ranges = ranges
    inline_tree = inline_parser.parse(content_bytes)

    query = Query(inline_lang, _LINK_QUERY)
    for _pattern_idx, captures in QueryCursor(query).matches(inline_tree.root_node):
        for dest_node in captures.get("dest", []):
            if dest_node.text is None:
                continue
            dest = dest_node.text.decode("utf-8", errors="replace")

            # Skip external URLs and fragment-only anchors.
            if any(dest.startswith(s) for s in _EXTERNAL_SCHEMES) or dest.startswith("#"):
                continue

            # Split path#fragment.
            module = dest
            names = ""
            if "#" in dest:
                module, names = dest.rsplit("#", 1)

            yield Chunk(
                id=make_chunk_id(file_path, blob_sha, f"link:{dest}", dest_node.start_point[0]),
                blob_sha=blob_sha,
                file_path=file_path,
                kind=ChunkKind.IMPORT,
                name=dest,
                scope="",
                content=dest,
                metadata=ImportMeta(module=module, names=names),
                line_start=dest_node.start_point[0] + 1,
                line_end=dest_node.end_point[0] + 1,
            )


# ── Plugin ───────────────────────────────────────────────────────────


markdown = LanguageRegistration(
    id="markdown",
    extensions=frozenset({".md"}),
    grammar_module="tree_sitter_markdown",
    injection_query=load_query(__package__, "injections"),
    language_plugin_version=4,
)

markdown.chunker(chunk_markdown)
