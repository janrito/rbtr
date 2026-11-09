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

import re
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from tree_sitter import Language, Parser, Query, QueryCursor

from rbtr.domain.models import Chunk, ChunkKind, Import, ImportMeta, Section, SectionMeta, Span
from rbtr.languages.chunks import chunk_plaintext, last_line
from rbtr.languages.registration import LanguageRegistration, load_query
from rbtr_lang_markdown.anchors import heading_anchors

if TYPE_CHECKING:
    from tree_sitter import Node, Range

# URL schemes that indicate external links (not cross-references).
_EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "ftp://")

# Block-level query: captures headed sections.
_SECTION_QUERY = load_query(__package__, "sections")

# Inline syntax a renderer does not show as a heading's text: code and
# emphasis delimiters, a link's destination and title, and HTML tags.
_UNRENDERED = frozenset(
    {"code_span_delimiter", "emphasis_delimiter", "link_destination", "link_title", "html_tag"}
)
_LINK_PUNCTUATION = frozenset({"[", "]", "(", ")", "!"})

# An `<a>` tag that sets an anchor with `id` or `name`.
_ANCHOR_TAG = re.compile(r"<a\b[^>]*?\b(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

# Injection query: a fenced code block delegates its content to the language
# named in the fence's info string (```python, ```sh). The info-string name
# is captured dynamically and resolved to a language id by the runner.


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


def _rendered_text(inline_parser: Parser, raw: bytes) -> str:
    """A heading's text as a renderer shows it: code without backticks, a link as its text."""
    tree = inline_parser.parse(raw)
    hidden: list[tuple[int, int]] = []
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        link_part = node.parent is not None and node.parent.type.endswith(("link", "image"))
        if node.type in _UNRENDERED or (link_part and node.type in _LINK_PUNCTUATION):
            hidden.append((node.start_byte, node.end_byte))
        else:
            pending.extend(node.children)
    shown = bytearray()
    position = 0
    for start, end in sorted(hidden):
        shown += raw[position:start]
        position = max(position, end)
    shown += raw[position:]
    return shown.decode("utf-8", errors="replace").strip()


def _set_anchors(section: Node) -> list[str]:
    """The anchors an `<a id>` or `<a name>` sets in the section's own paragraphs and HTML."""
    own = []
    for child in section.children:
        if child.type == section.type:
            break
        if child.type in ("paragraph", "html_block") and child.text is not None:
            own.extend(_ANCHOR_TAG.findall(child.text.decode("utf-8", errors="replace")))
    return own


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

    **Anchors** are what each common renderer makes of the heading
    (`heading_anchors`), and any the section sets itself with `<a id>` or
    `<a name>`.
    """
    import tree_sitter_markdown  # deferred: heavy native lib

    inline_parser = Parser(Language(tree_sitter_markdown.inline_language()))
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

    headed = [
        (capture_dict["doc_section"][0], capture_dict["_section_name"][0])
        for _pattern_idx, capture_dict in matches
        if capture_dict.get("doc_section") and capture_dict.get("_section_name")
    ]
    anchors = heading_anchors(
        [_rendered_text(inline_parser, heading.text or b"") for _node, heading in headed]
    )

    for (node, heading), rendered_anchors in zip(headed, anchors, strict=True):
        name = heading.text.decode("utf-8", errors="replace") if heading.text else ""

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
        section_anchors = tuple(dict.fromkeys((*rendered_anchors, *_set_anchors(node))))
        yield Chunk.model_validate(
            {
                "blob_sha": blob_sha,
                "file_path": file_path,
                "role": Section(
                    kind=ChunkKind.DOC_SECTION,
                    name=name,
                    meta=SectionMeta(anchors=section_anchors),
                ),
                "scope": scope_segments,
                "language": "markdown",
                "content": text,
                "span": Span(start=line_start, end=last_line(node)),
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


_LINK_QUERY = load_query(__package__, "links")


def _extract_links(
    content_bytes: bytes,
    file_path: str,
    blob_sha: str,
    ranges: list[Range] | None,
) -> Iterator[Chunk]:
    """Extract local links as IMPORT chunks using the inline parser.

    Parses the full content with `tree_sitter_markdown.inline_language()`
    and queries for `inline_link` nodes.  External URLs are skipped; a
    fragment alone (`#usage`) names an anchor in the document itself.
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

            # Skip external URLs.
            if any(dest.startswith(s) for s in _EXTERNAL_SCHEMES):
                continue

            # Split path#fragment; a fragment alone names an anchor in this document.
            module = dest
            names = ""
            if "#" in dest:
                module, names = dest.rsplit("#", 1)
            if not module:
                module = PurePosixPath(file_path).name

            yield Chunk(
                blob_sha=blob_sha,
                file_path=file_path,
                role=Import(name=dest, meta=ImportMeta(module=module, names=names)),
                content=dest,
                span=Span(start=dest_node.start_point[0] + 1, end=last_line(dest_node)),
            )


markdown = LanguageRegistration(
    id="markdown",
    extensions=frozenset({".md"}),
    grammar_module="tree_sitter_markdown",
    injection_query=load_query(__package__, "injections"),
    prose=True,
    extraction_serial=6,
)

markdown.chunker(chunk_markdown)
