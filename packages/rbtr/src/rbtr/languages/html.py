"""HTML language plugin.

Splits HTML by major structural elements inside `<body>` and
extracts cross-language import references from `<head>`.

Extracted chunks::

    <head>
      <script src="app.js">         → import, metadata {module: "app.js"}
      <link href="styles.css">      → import, metadata {module: "styles.css"}
    </head>
    <body>
      <h1>Title</h1>                → doc_section "h1", scope ""
      <p>Content</p>                → doc_section "p", scope ""
    </body>
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Parser

from rbtr.index.chunks import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind, ImportMeta
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Language, Node

# ── Chunking ─────────────────────────────────────────────────────────


def _find_element(node: Node, tag_name: str) -> Node | None:
    """Find a named element in the HTML tree."""
    if node.type == "element":
        for child in node.children:
            if child.type == "start_tag":
                for gc in child.children:
                    if gc.type == "tag_name" and gc.text == tag_name.encode():
                        return node
    for child in node.children:
        result = _find_element(child, tag_name)
        if result is not None:
            return result
    return None


def _get_attr(start_tag: Node, attr_name: str) -> str | None:
    """Extract an attribute value from a start_tag node."""
    for child in start_tag.children:
        if child.type == "attribute":
            name_node = None
            value_node = None
            for gc in child.children:
                if gc.type == "attribute_name" and gc.text:
                    name_node = gc
                elif gc.type == "quoted_attribute_value":
                    for vg in gc.children:
                        if vg.type == "attribute_value" and vg.text:
                            value_node = vg
            if (
                name_node is not None
                and value_node is not None
                and name_node.text == attr_name.encode()
            ):
                text = value_node.text
                return text.decode("utf-8", errors="replace") if text else None
    return None


def _extract_imports(
    node: Node,
    content_bytes: bytes,
    file_path: str,
    blob_sha: str,
) -> Iterator[Chunk]:
    """Walk the tree and yield <script src> and <link href> as import chunks."""
    for child in node.children:
        start_tag = None
        if child.type == "script_element":
            for gc in child.children:
                if gc.type == "start_tag":
                    start_tag = gc
                    break
            if start_tag is not None:
                src = _get_attr(start_tag, "src")
                if src:
                    text = (
                        content_bytes[child.start_byte : child.end_byte]
                        .decode("utf-8", errors="replace")
                        .strip()
                    )
                    yield Chunk(
                        id=make_chunk_id(
                            file_path, blob_sha, f"script:{src}", child.start_point[0]
                        ),
                        blob_sha=blob_sha,
                        file_path=file_path,
                        kind=ChunkKind.IMPORT,
                        name=text,
                        scope="",
                        content=text,
                        metadata=ImportMeta(module=src, language_hint="javascript"),
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                    )
        elif child.type == "element":
            for gc in child.children:
                if gc.type == "start_tag":
                    start_tag = gc
                    break
            if start_tag is not None:
                for gc in start_tag.children:
                    if gc.type == "tag_name" and gc.text == b"link":
                        href = _get_attr(start_tag, "href")
                        if href:
                            text = (
                                content_bytes[child.start_byte : child.end_byte]
                                .decode("utf-8", errors="replace")
                                .strip()
                            )
                            yield Chunk(
                                id=make_chunk_id(
                                    file_path, blob_sha, f"link:{href}", child.start_point[0]
                                ),
                                blob_sha=blob_sha,
                                file_path=file_path,
                                kind=ChunkKind.IMPORT,
                                name=text,
                                scope="",
                                content=text,
                                metadata=ImportMeta(module=href, language_hint="css"),
                                line_start=child.start_point[0] + 1,
                                line_end=child.end_point[0] + 1,
                            )
                        break
        # Recurse into nested elements.
        if child.is_named:
            yield from _extract_imports(child, content_bytes, file_path, blob_sha)


def chunk_html(file_path: str, blob_sha: str, content: str, grammar: Language) -> Iterator[Chunk]:
    """Split HTML by major structural elements and extract imports.

    Produces `DOC_SECTION` chunks from `<body>` children and
    `IMPORT` chunks from `<script src>` and `<link href>` in
    `<head>`. Falls back to a single chunk if no `<body>` is found.
    """
    if not content.strip():
        return
    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    tree = parser.parse(content_bytes)

    # Extract imports from the entire document (script/link in head).
    import_chunks: list[Chunk] = list(
        _extract_imports(tree.root_node, content_bytes, file_path, blob_sha)
    )

    # Extract body elements as doc sections.
    body = _find_element(tree.root_node, "body")
    if body is None and not import_chunks:
        yield Chunk(
            id=make_chunk_id(file_path, blob_sha, file_path, 0),
            blob_sha=blob_sha,
            file_path=file_path,
            kind=ChunkKind.DOC_SECTION,
            name=file_path,
            scope="",
            content=content.strip(),
            line_start=1,
            line_end=tree.root_node.end_point[0] + 1,
        )
        return

    yield from import_chunks

    if body is not None:
        for child in body.children:
            if child.type == "element":
                tag = ""
                for gc in child.children:
                    if gc.type == "start_tag":
                        for tg in gc.children:
                            if tg.type == "tag_name" and tg.text:
                                tag = tg.text.decode("utf-8", errors="replace")
                                break
                        break
                text = (
                    content_bytes[child.start_byte : child.end_byte]
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                if text:
                    yield Chunk(
                        id=make_chunk_id(file_path, blob_sha, tag, child.start_point[0]),
                        blob_sha=blob_sha,
                        file_path=file_path,
                        kind=ChunkKind.DOC_SECTION,
                        name=tag,
                        scope="",
                        content=text,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                    )


# ── Plugin ───────────────────────────────────────────────────────────


class HtmlPlugin:
    """HTML language support — body-element chunking + cross-language imports."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="html",
                extensions=frozenset({".html", ".htm"}),
                grammar_module="tree_sitter_html",
                chunker=chunk_html,
                import_targets=frozenset({"javascript", "typescript", "css"}),
            ),
        ]
