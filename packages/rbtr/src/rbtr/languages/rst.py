"""RST language plugin.

Splits reStructuredText by heading hierarchy using tree-sitter.
RST sections are flat in the AST (title + adornment only), so
the chunker reconstructs hierarchy from adornment character
order (first seen = level 1, second = level 2, etc.).

Extracted chunks::

    Title                           → doc_section "Title", scope ""
    =====                             (content: heading through next heading)

    Section A                       → doc_section "Section A", scope "Title"
    ---------                         (adornment char "-" = level 2)

    First paragraph.                → doc_section "", scope ""  (no heading)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Parser, Query, QueryCursor

from rbtr.index.identity import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind, ImportMeta
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Language, Node

# URL schemes that indicate external links.
_EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "ftp://")

# RST roles that reference a code symbol by name.
_SYMBOL_ROLES = frozenset({":func:", ":class:", ":meth:", ":attr:"})
# RST roles that reference a document/module by path.
_PATH_ROLES = frozenset({":doc:", ":mod:"})

# ── Chunking ─────────────────────────────────────────────────────────


def _rst_title_text(section: Node) -> str:
    """Extract the title text from an RST section node."""
    for child in section.children:
        if child.type == "title" and child.text:
            return child.text.decode("utf-8", errors="replace")
    return ""


def _rst_adornment_char(section: Node) -> str:
    """Return the adornment character of an RST section."""
    for child in section.children:
        if child.type == "adornment" and child.text:
            return child.text.decode()[0]
    return ""


def chunk_rst(file_path: str, blob_sha: str, content: str, grammar: Language) -> Iterator[Chunk]:
    """Split RST by heading hierarchy using tree-sitter.

    Reconstructs the heading hierarchy from adornment characters
    (first character seen = level 1, second = level 2, etc.).
    Each section gets content from its heading to the next
    same-or-higher-level heading.
    """
    if not content.strip():
        return

    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    tree = parser.parse(content_bytes)
    root = tree.root_node

    children = [c for c in root.children if c.is_named]

    has_sections = any(c.type == "section" for c in children)
    if not has_sections:
        for child in children:
            if child.type == "paragraph":
                text = (
                    content_bytes[child.start_byte : child.end_byte]
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                if text:
                    yield Chunk(
                        id=make_chunk_id(
                            file_path,
                            blob_sha,
                            f"para:{child.start_point[0]}",
                            child.start_point[0],
                        ),
                        blob_sha=blob_sha,
                        file_path=file_path,
                        kind=ChunkKind.DOC_SECTION,
                        name="",
                        scope="",
                        content=text,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0],
                    )
        return

    # Build heading level map from adornment character order.
    adornment_levels: dict[str, int] = {}
    for child in children:
        if child.type == "section":
            char = _rst_adornment_char(child)
            if char and char not in adornment_levels:
                adornment_levels[char] = len(adornment_levels) + 1

    # Walk siblings: associate content with preceding section.
    chunks: list[Chunk] = []
    scope_stack: list[tuple[str, int]] = []
    current_title = ""
    current_start = 0
    current_line_start = 1
    in_section = False

    for child in children:
        if child.type == "section":
            # Close previous section if open.
            if in_section:
                text = (
                    content_bytes[current_start : child.start_byte]
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                if text:
                    # Exclude the section being closed (the stack top) so
                    # scope is the enclosing path only — never self. The
                    # final-section close below does the same via [:-1].
                    chunks.append(
                        Chunk.model_validate(
                            {
                                "blob_sha": blob_sha,
                                "file_path": file_path,
                                "kind": ChunkKind.DOC_SECTION,
                                "name": current_title,
                                "scope": [t for t, _ in scope_stack[:-1]],
                                "content": text,
                                "line_start": current_line_start,
                                "line_end": child.start_point[0],
                            }
                        )
                    )

            # Start new section.
            title = _rst_title_text(child)
            char = _rst_adornment_char(child)
            level = adornment_levels.get(char, 0)

            # Pop scope stack to parent level.
            while scope_stack and scope_stack[-1][1] >= level:
                scope_stack.pop()

            current_title = title
            current_start = child.start_byte
            current_line_start = child.start_point[0] + 1
            in_section = True

            scope_stack.append((title, level))

    # Close final section and yield all accumulated chunks.
    if in_section:
        text = content_bytes[current_start:].decode("utf-8", errors="replace").strip()
        if text:
            scope_stack_for_scope = scope_stack[:-1]
            chunks.append(
                Chunk.model_validate(
                    {
                        "blob_sha": blob_sha,
                        "file_path": file_path,
                        "kind": ChunkKind.DOC_SECTION,
                        "name": current_title,
                        "scope": [t for t, _ in scope_stack_for_scope],
                        "content": text,
                        "line_start": current_line_start,
                        "line_end": root.end_point[0],
                    }
                )
            )

    yield from chunks

    # Extract cross-references as IMPORT chunks.
    yield from _extract_references(root, grammar, content_bytes, file_path, blob_sha)


# ── Reference extraction ─────────────────────────────────────────────

_REF_QUERY = """\
(interpreted_text (role) @role) @ref

(directive (type) @dir_type (body (content) @content)) @directive

(reference) @hyperlink
"""


def _strip_rst_target(role: str, raw: str) -> str:
    """Clean an RST role target: strip backticks and ~ prefix.

    For symbol roles (:func:, :class:, :meth:), `~` means
    "show only the last component" — we extract the last
    component as the symbol name.
    """
    target = raw.strip("`")
    if target.startswith("~"):
        target = target[1:]
        # ~module.Class.method → method (last component)
        if "." in target:
            target = target.rsplit(".", 1)[1]
    return target


def _extract_references(
    root: Node,
    grammar: Language,
    content_bytes: bytes,
    file_path: str,
    blob_sha: str,
) -> Iterator[Chunk]:
    """Extract RST cross-references as IMPORT chunks.

    Uses tree-sitter queries on the already-parsed tree for:
    - Domain roles (:func:, :class:, :doc:, etc.)
    - Toctree directives
    - Hyperlink references (`text <url>`_)
    """
    query = Query(grammar, _REF_QUERY)
    for _, captures in QueryCursor(query).matches(root):
        # Domain roles: :func:`name`, :doc:`path`, etc.
        if "ref" in captures:
            role_nodes = captures.get("role", [])
            ref_nodes = captures["ref"]
            if not role_nodes or not ref_nodes:
                continue
            role_text = role_nodes[0].text.decode() if role_nodes[0].text else ""
            ref_node = ref_nodes[0]

            # Get inner interpreted_text child (the target).
            target_text = ""
            for child in ref_node.children:
                if child.type == "interpreted_text" and child is not role_nodes[0]:
                    target_text = child.text.decode().strip("`") if child.text else ""
                    break
            if not target_text:
                continue

            target = _strip_rst_target(role_text, target_text)
            meta = ImportMeta()
            if role_text in _SYMBOL_ROLES:
                meta.names = target
            elif role_text in _PATH_ROLES:
                meta.module = target
            else:
                continue

            yield Chunk(
                id=make_chunk_id(
                    file_path, blob_sha, f"role:{role_text}{target}", ref_node.start_point[0]
                ),
                blob_sha=blob_sha,
                file_path=file_path,
                kind=ChunkKind.IMPORT,
                name=f"{role_text}`{target_text}`",
                scope="",
                content=ref_node.text.decode() if ref_node.text else "",
                metadata=meta,
                line_start=ref_node.start_point[0] + 1,
                line_end=ref_node.end_point[0] + 1,
            )

        # Toctree directives.
        elif "directive" in captures:
            type_nodes = captures.get("dir_type", [])
            content_nodes = captures.get("content", [])
            if not type_nodes or not content_nodes:
                continue
            dtype = type_nodes[0].text.decode() if type_nodes[0].text else ""
            if dtype != "toctree":
                continue
            raw_content = content_nodes[0].text.decode() if content_nodes[0].text else ""
            for line in raw_content.splitlines():
                entry = line.strip()
                if not entry:
                    continue
                yield Chunk(
                    id=make_chunk_id(
                        file_path, blob_sha, f"toctree:{entry}", content_nodes[0].start_point[0]
                    ),
                    blob_sha=blob_sha,
                    file_path=file_path,
                    kind=ChunkKind.IMPORT,
                    name=f"toctree:{entry}",
                    scope="",
                    content=entry,
                    metadata=ImportMeta(module=entry),
                    line_start=content_nodes[0].start_point[0] + 1,
                    line_end=content_nodes[0].end_point[0] + 1,
                )

        # Hyperlink references: `text <url>`_
        elif "hyperlink" in captures:
            for ref_node in captures["hyperlink"]:
                if ref_node.text is None:
                    continue
                text = ref_node.text.decode()
                # Extract URL from between < and > in node text.
                lt = text.rfind("<")
                gt = text.rfind(">")
                if lt == -1 or gt == -1 or gt <= lt:
                    continue
                url = text[lt + 1 : gt]
                if any(url.startswith(s) for s in _EXTERNAL_SCHEMES):
                    continue
                yield Chunk(
                    id=make_chunk_id(file_path, blob_sha, f"ref:{url}", ref_node.start_point[0]),
                    blob_sha=blob_sha,
                    file_path=file_path,
                    kind=ChunkKind.IMPORT,
                    name=text,
                    scope="",
                    content=text,
                    metadata=ImportMeta(module=url),
                    line_start=ref_node.start_point[0] + 1,
                    line_end=ref_node.end_point[0] + 1,
                )


# ── Plugin ───────────────────────────────────────────────────────────


class RstPlugin:
    """RST language support — heading-hierarchy chunking."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="rst",
                extensions=frozenset({".rst"}),
                grammar_module="tree_sitter_rst",
                chunker=chunk_rst,
                language_plugin_version=3,
            ),
        ]
